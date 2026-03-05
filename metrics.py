import numpy as np
from surface_distance import metrics as sd_metrics
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.metrics import normalized_mutual_info_score # Ajout pour le NMI

from apsl_mask import skeleton_to_graph_sampled
from apls import APLSMetric

EPSILON = 1e-8


def cl_score(v, s):
    """Calcule le cl_score basé sur cldice_metric.py"""
    return float(np.sum(v * s)) / float(np.sum(s) + EPSILON)


def clDice(v_p, v_l):
    """Calcule le clDice basé sur cldice_metric.py"""
    skel_l = skeletonize(v_l)
    skel_p = skeletonize(v_p)
    tprec = cl_score(v_p, skel_l)
    tsens = cl_score(v_l, skel_p)
    return 2.0 * tprec * tsens / (tprec + tsens + EPSILON)


def compute_advanced_metrics(pred_mask, gt_mask, do_heavy):
    """
    Fonction principale assemblant toutes les sous-métriques.
    Argument 'debug=True' pour afficher les étapes.
    """
    y_pred = pred_mask.astype(bool)
    y_true = gt_mask.astype(bool)

    results = {}

    results.update(compute_pixel_metrics(y_pred, y_true))
    
    results['cldice'] = clDice(y_pred, y_true)

    pred_sum = np.count_nonzero(y_pred)
    true_sum = np.count_nonzero(y_true)
    if pred_sum > 0 and true_sum > 0:
        try:
            surface_distances = sd_metrics.compute_surface_distances(
                y_true, y_pred, spacing_mm=(0.0487, 0.0487)
            )


            results['hausdorff_95'] = sd_metrics.compute_robust_hausdorff(surface_distances, 95)
            results['hausdorff_max'] = sd_metrics.compute_robust_hausdorff(surface_distances, 100)
            results['surface_dice_1mm'] = sd_metrics.compute_surface_dice_at_tolerance(surface_distances, 1.0)

        except Exception as e:
            print(f"Erreur Surface Distance: {e}")
            results['hausdorff_95'] = -1.0
            results['hausdorff_max'] = -1.0
            results['hausdorff_95'] = -1.0
            results['hausdorff_max'] = -1.0
            results['surface_dice_1mm'] = -1.0
    else:
        val = 0.0 if (pred_sum == 0 and true_sum == 0) else float('inf')
        results['hausdorff_distance95'] = val
        results['hausdorff_distance'] = val
        results['hausdorff_95'] = val
        results['hausdorff_max'] = val
        results['surface_dice_1mm'] = 0.0

    results.update(compute_topology_metrics(y_pred, y_true))

    results['ASCD'] = compute_centerline_metric(y_pred, y_true)

    if do_heavy:
        labeled_true = label(y_true, connectivity=2)
        apls_scores = []
        apls_recalls = []
        apls_precisions = []
        for region in regionprops(labeled_true):
            minr, minc, maxr, maxc = region.bbox
            y_true_cc = (labeled_true[minr:maxr, minc:maxc] == region.label)
            y_pred_cc = y_pred[minr:maxr, minc:maxc]
            apls_result = compute_apls_metric(y_pred_cc, y_true_cc, snap_px=5)
            apls_scores.append(float(apls_result['apls']))
            apls_recalls.append(float(apls_result['apls_recall']))
            apls_precisions.append(float(apls_result['apls_precision']))

        results.update({
            'apls': float(np.mean(apls_scores)) if len(apls_scores) > 0 else 0.0,
            'apls_recall': float(np.mean(apls_recalls)) if len(apls_recalls) > 0 else 0.0,
            'apls_precision': float(np.mean(apls_precisions)) if len(apls_precisions) > 0 else 0.0
        })

    return results


def compute_pixel_metrics(y_pred, y_true):
    intersection = float(np.logical_and(y_pred, y_true).sum())
    
    tp = intersection
    fp = float(np.logical_and(y_pred, ~y_true).sum())
    fn = float(np.logical_and(~y_pred, y_true).sum())
    tn = float(np.logical_and(~y_pred, ~y_true).sum())

    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1_score = 2.0 * (precision * recall) / (precision + recall + EPSILON)

    def f_beta(beta):
        return (1 + (beta**2)) * (precision * recall) / (((beta**2) * precision) + recall + EPSILON)

    iou_fg = tp / (tp + fp + fn + EPSILON)
    iou_bg = tn / (tn + fp + fn + EPSILON)
    mean_iou = (iou_fg + iou_bg) / 2.0

    pixel_acc = (tp + tn) / (tp + tn + fp + fn + EPSILON)
    specificity = tn / (tn + fp + EPSILON)
    
    dice = sd_metrics.compute_dice_coefficient(y_true, y_pred)
    
    nmi = normalized_mutual_info_score(y_true.flatten(), y_pred.flatten(), average_method='geometric')

    return {
        'dice': dice,
        'f1_score': f1_score,
        'f_2_score': f_beta(2.0),
        'f_3_score': f_beta(3.0),
        'f_4_score': f_beta(4.0),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'pixel_accuracy': pixel_acc,
        'iou': iou_fg,
        'mean_iou': mean_iou,
        'normalized_mutual_information': nmi
    }


def compute_centerline_metric(y_pred, y_true, pixel_size=0.0487):
    # (Identique à l'original)
    pred_sum = np.count_nonzero(y_pred)
    true_sum = np.count_nonzero(y_true)

    if pred_sum == 0 and true_sum == 0:
        return 0.0
    elif pred_sum == 0 or true_sum == 0:
        return float('inf')

    skel_pred = skeletonize(y_pred.astype(bool))
    skel_gt = skeletonize(y_true.astype(bool))

    dist_map_gt = distance_transform_edt(np.logical_not(skel_gt))
    dist_map_pred = distance_transform_edt(np.logical_not(skel_pred))

    d1 = dist_map_gt[skel_pred]
    d2 = dist_map_pred[skel_gt]
    
    d1 = d1 * pixel_size
    d2 = d2 * pixel_size
    
    if d1.size == 0 and d2.size == 0:
        return 0.0 

    all_d = np.concatenate([d1, d2])
    
    return float(np.mean(all_d))


def _get_betti_numbers(binary_img):
    labeled_img = label(binary_img, connectivity=2)
    regions = regionprops(labeled_img)
    betti_0 = float(len(regions))
    euler_char = float(np.sum([region.euler_number for region in regions]))
    betti_1 = float(betti_0 - euler_char)  # β1 = β0 - χ
    return betti_0, betti_1


def compute_topology_metrics(y_pred, y_true):
    b0_pred, b1_pred = _get_betti_numbers(y_pred)
    b0_gt, b1_gt = _get_betti_numbers(y_true)

    betti_0_abs_error = abs(b0_pred - b0_gt)
    betti_1_abs_error = abs(b1_pred - b1_gt)

    b0_jaccard = min(b0_pred, b0_gt) / (max(b0_pred, b0_gt) + EPSILON)
    b0_rel_err = abs(b0_pred - b0_gt) / (b0_gt + EPSILON)
    b0_var_idx = abs(b0_pred - b0_gt) / (b0_pred + b0_gt + EPSILON)

    b1_jaccard = min(b1_pred, b1_gt) / (max(b1_pred, b1_gt) + EPSILON)
    b1_rel_err = abs(b1_pred - b1_gt) / (b1_gt + EPSILON)
    b1_var_idx = abs(b1_pred - b1_gt) / (b1_pred + b1_gt + EPSILON)

    return {
        'betti_0_abs_err': betti_0_abs_error,
        'betti_1_abs_err': betti_1_abs_error,
        'b0_pred': b0_pred,
        'b1_pred': b1_pred,

        'betti0_jaccard_ratio': b0_jaccard,
        'betti0_relative_error': b0_rel_err,
        'betti0_variation_index': b0_var_idx,

        'betti1_jaccard_ratio': b1_jaccard,
        'betti1_relative_error': b1_rel_err,
        'betti1_variation_index': b1_var_idx,
    }


def compute_apls_metric(y_pred, y_true, snap_px=4):
    pred_sum = np.count_nonzero(y_pred)
    true_sum = np.count_nonzero(y_true)

    if true_sum == 0:
        val = 1.0 if pred_sum == 0 else 0.0
        return {'apls': val, 'apls_recall': val, 'apls_precision': val}

    if pred_sum == 0:
        return {'apls': 0.0, 'apls_recall': 0.0, 'apls_precision': 0.0}

    try:
        if not y_true.flags['C_CONTIGUOUS']: y_true = np.ascontiguousarray(y_true)
        if not y_pred.flags['C_CONTIGUOUS']: y_pred = np.ascontiguousarray(y_pred)

        skel_gt = skeletonize(y_true)
        skel_pred = skeletonize(y_pred)

        if np.count_nonzero(skel_gt) == 0:
            return {'apls': 0.0, 'apls_recall': 0.0, 'apls_precision': 0.0}
        if np.count_nonzero(skel_pred) == 0:
            return {'apls': 0.0, 'apls_recall': 0.0, 'apls_precision': 0.0}

    except Exception as e:
        print(f"Skel error: {e}")
        return {'apls': 0.0, 'apls_recall': 0.0, 'apls_precision': 0.0}

    try:
        G_gt = skeleton_to_graph_sampled(skel_gt, sample_dist=20.0)
        G_pred = skeleton_to_graph_sampled(skel_pred, sample_dist=20.0)
    except Exception as e:
        print(f"Skel2Graph error: {e}")
        return {'apls': 0.0, 'apls_recall': 0.0, 'apls_precision': 0.0}

    try:
        MAX_APLS_TIME = 60 * 10
        metric = APLSMetric(G_gt, G_pred, snap_buffer_meters=float(snap_px), max_time=None)
        score = metric.compute()

        return {
            'apls': score['f1'],
            'apls_recall': score['recall'],
            'apls_precision': score['precision']
        }

    except TimeoutError:
        print(f"⚠️ APLS calculation timed out after {MAX_APLS_TIME}s. Skipping.")
        return {'apls': -1.0, 'apls_recall': -1.0, 'apls_precision': -1.0}

    except Exception as e:
        print(f"Erreur calcul APLS: {e}")
        return {'apls': 0.0, 'apls_recall': 0.0, 'apls_precision': 0.0}