import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

EPSILON = 1e-8

def compute_advanced_metrics(pred_mask, gt_mask):
    """
    Fonction principale assemblant toutes les sous-métriques.
    Argument 'debug=True' pour afficher les étapes.
    """
    # Conversion booléenne une seule fois
    y_pred = pred_mask.astype(bool)
    y_true = gt_mask.astype(bool)

    results = {}

    # 1. Pixel Metrics
    results.update(compute_pixel_metrics(y_pred, y_true))

    # 2. Hausdorff (Souvent lourd, on peut l'isoler dans un try/except)
    try:
        results['hausdorff'] = compute_hausdorff_metric(y_pred, y_true)
    except Exception as e:
        print(f"Erreur Hausdorff: {e}")
        results['hausdorff'] = -1.0

    # 3. Topology
    results.update(compute_topology_metrics(y_pred, y_true))

    # 4. Centerline
    results['centerline_dist'] = compute_centerline_metric(y_pred, y_true)

    return results
   
def compute_pixel_metrics(y_pred, y_true):
    """Calcule Precision, Recall, F1, IoU, Dice."""
    intersection = float(np.logical_and(y_pred, y_true).sum())
    union = float(np.logical_or(y_pred, y_true).sum())
    
    tp = intersection
    fp = float(np.logical_and(y_pred, ~y_true).sum())
    fn = float(np.logical_and(~y_pred, y_true).sum())
    
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1_score = 2.0 * (precision * recall) / (precision + recall + EPSILON)
    
    iou = intersection / (union + EPSILON)
    dice = 2.0 * intersection / (y_pred.sum() + y_true.sum() + EPSILON)
    
    return {
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'dice': dice
    }
    
def compute_hausdorff_metric(y_pred, y_true):
    """Calcule la distance de Hausdorff."""
    if float(y_pred.sum()) > 0 and float(y_true.sum()) > 0:
        coords_pred = np.argwhere(y_pred)
        coords_gt = np.argwhere(y_true)
        
        d_pred_gt = directed_hausdorff(coords_pred, coords_gt)[0]
        d_gt_pred = directed_hausdorff(coords_gt, coords_pred)[0]
        return float(max(d_pred_gt, d_gt_pred))
    else:
        # Pénalité max si un masque est vide alors que l'autre non, ou 0 si les deux vides
        return 0.0 if (float(y_pred.sum()) == 0 and float(y_true.sum()) == 0) else 100.0
    
    
def compute_centerline_metric(y_pred, y_true):
    """Calcule la distance moyenne entre les squelettes."""
    if y_pred.sum() == 0 or y_true.sum() == 0:
        return 0.0

    skel_pred = skeletonize(y_pred)
    skel_gt = skeletonize(y_true)
    
    # Carte de distance inversée
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
        
    return (acl_pred_to_gt + acl_gt_to_pred) / 2.0
    
def _get_betti_numbers(binary_img):
    """Fonction helper pour extraire Betti 0 et Betti 1."""
    labeled_img = label(binary_img)
    regions = regionprops(labeled_img)
    betti_0 = float(len(regions))  
    euler_char = float(np.sum([region.euler_number for region in regions]))
    betti_1 = float(betti_0 - euler_char)  # β1 = β0 - χ
    return betti_0, betti_1

def compute_topology_metrics(y_pred, y_true):
    """Calcule l'erreur sur les nombres de Betti."""
    b0_pred, b1_pred = _get_betti_numbers(y_pred)
    b0_gt, b1_gt = _get_betti_numbers(y_true)
    
    betti_0_error = abs(b0_pred - b0_gt) / (b0_pred + b0_gt + EPSILON)
    betti_1_error = abs(b1_pred - b1_gt) / (b1_pred + b1_gt + EPSILON)
    
    return {
        'betti_0_err': betti_0_error,
        'betti_1_err': betti_1_error,
        'b0_pred': b0_pred,
        'b1_pred': b1_pred
    }