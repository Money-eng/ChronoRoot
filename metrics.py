import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, euler_number
from sklearn.metrics import precision_recall_fscore_support

class RootMetrics:
    """
    Wrapper utilisant scipy, skimage et sklearn pour calculer les métriques
    topologiques et de segmentation sur des images complètes.
    """

    @staticmethod
    def _prepare(mask):
        """Assure que le masque est booléen pour les libs."""
        return (mask > 0.5).astype(bool)

    @staticmethod
    def compute_all(y_true, y_pred):
        """
        Calcule toutes les métriques d'un coup.
        y_true, y_pred: Arrays 2D (H, W)
        """
        gt = RootMetrics._prepare(y_true)
        pred = RootMetrics._prepare(y_pred)
        
        # Si l'une des images est vide, on retourne des valeurs par défaut
        if not np.any(gt) or not np.any(pred):
            return {
                "dice": 0.0, "precision": 0.0, "recall": 0.0,
                "hausdorff": -1.0, # Indicateur d'erreur
                "cl_dice": 0.0, "betti_0_error": 0, "betti_1_error": 0
            }

        # --- 1. Pixel Metrics (sklearn) ---
        # average='binary' traite la classe True comme la cible
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt.flatten(), pred.flatten(), average='binary', zero_division=0
        )
        
        # --- 2. Distance de Hausdorff (scipy) ---
        # directed_hausdorff calcule max(min(d(u, v))). La distance symétrique est le max des deux sens.
        # On passe les coordonnées des points non-nuls via np.argwhere
        # u_points = np.argwhere(gt)
        # v_points = np.argwhere(pred)
        # forward = directed_hausdorff(u_points, v_points)[0]
        # backward = directed_hausdorff(v_points, u_points)[0]
        # hausdorff = max(forward, backward)

        # --- 3. Topologie / Betti Numbers (skimage) ---
        # Betti 0 (Composantes connexes)
        _, n_comps_gt = label(gt, return_num=True, connectivity=2)
        _, n_comps_pred = label(pred, return_num=True, connectivity=2)
        
        # Betti 1 (Trous/Cycles) via Euler Number (Chi = B0 - B1 => B1 = B0 - Chi)
        euler_gt = euler_number(gt, connectivity=2)
        euler_pred = euler_number(pred, connectivity=2)
        betti_1_gt = n_comps_gt - euler_gt
        betti_1_pred = n_comps_pred - euler_pred

        # --- 4. Centerline Dice (clDice) - "Metric sur squelette" ---
        # Utilise skimage.morphology.skeletonize
        skel_true = skeletonize(gt)
        skel_pred = skeletonize(pred)
        
        # Tprec: Fraction du squelette prédit qui touche le masque GT
        tprec = np.sum(skel_pred * gt) / (np.sum(skel_pred) + 1e-6)
        # Tsens: Fraction du squelette GT qui touche le masque prédit
        tsens = np.sum(skel_true * pred) / (np.sum(skel_true) + 1e-6)
        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-6)

        return {
            "dice": f1,           # F1 est équivalent au Dice en binaire
            "precision": precision,
            "recall": recall,
            #"hausdorff": hausdorff,
            "cl_dice": cl_dice,
            "betti_0_error": abs(n_comps_gt - n_comps_pred) / (n_comps_gt +n_comps_gt + 1e-8),
            "betti_1_error": abs(betti_1_gt - betti_1_pred) / (betti_1_gt + betti_1_pred + 1e-8)
        }