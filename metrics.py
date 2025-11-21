import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

def compute_advanced_metrics(pred_mask, gt_mask):
    """
    Calcule un ensemble complet de métriques pour la segmentation binaire.
    Entrées :
        pred_mask : numpy array 2D (H, W), binaire (0 ou 1)
        gt_mask   : numpy array 2D (H, W), binaire (0 ou 1)
    """

    y_pred = pred_mask.astype(bool)
    y_true = gt_mask.astype(bool)

    intersection = float(np.logical_and(y_pred, y_true).sum())
    union = float(np.logical_or(y_pred, y_true).sum())
    
    tp = intersection
    fp = float(np.logical_and(y_pred, ~y_true).sum())
    fn = float(np.logical_and(~y_pred, y_true).sum())
    
    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = 2.0 * (precision * recall) / (precision + recall + epsilon)
    
    iou = intersection / (union + epsilon)
    dice = 2.0 * intersection / (y_pred.sum() + y_true.sum() + epsilon)

    if float(y_pred.sum()) > 0 and float(y_true.sum()) > 0:
        coords_pred = np.argwhere(y_pred)
        coords_gt = np.argwhere(y_true)
        
        d_pred_gt = directed_hausdorff(coords_pred, coords_gt)[0]
        d_gt_pred = directed_hausdorff(coords_gt, coords_pred)[0]
        hausdorff_dist = float(max(d_pred_gt, d_gt_pred))
    else:
        hausdorff_dist = 0.0 if (float(y_pred.sum()) == 0 and float(y_true.sum()) == 0) else 100.0
    
    def get_betti_numbers(binary_img):
        labeled_img = label(binary_img)
        regions = regionprops(labeled_img)
        betti_0 = float(len(regions))  
        euler_char = float(np.sum([region.euler_number for region in regions]))
        betti_1 = float(betti_0 - euler_char)  # β1 = β0 - χ
        return betti_0, betti_1

    b0_pred, b1_pred = get_betti_numbers(y_pred)
    b0_gt, b1_gt = get_betti_numbers(y_true)
    
    betti_0_error = abs(b0_pred - b0_gt) / (b0_pred + b0_gt + epsilon)
    betti_1_error = abs(b1_pred - b1_gt) / (b1_pred + b1_gt + epsilon)


    if y_pred.sum() > 0 and y_true.sum() > 0:
        skel_pred = skeletonize(y_pred)
        skel_gt = skeletonize(y_true)
        
        dist_map_gt = distance_transform_edt(np.logical_not(skel_gt))
        dist_map_pred = distance_transform_edt(np.logical_not(skel_pred))
        
        if np.sum(skel_pred) > 0:
            acl_pred_to_gt = float(np.mean(dist_map_gt[skel_pred]))
        else:
            acl_pred_to_gt = 0.0
            
        if np.sum(skel_gt) > 0:
            acl_gt_to_pred = float(np.mean(dist_map_pred[skel_gt]))
        else:
            acl_gt_to_pred = 0.0
            
        avg_centerline_dist = (acl_pred_to_gt + acl_gt_to_pred) / 2.0
    else:
        avg_centerline_dist = 0.0

    return {
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'dice': dice,
        'hausdorff': hausdorff_dist,
        'betti_0_err': betti_0_error,
        'betti_1_err': betti_1_error,
        'b0_pred': b0_pred, 
        'b1_pred': b1_pred,
        'centerline_dist': avg_centerline_dist
    }