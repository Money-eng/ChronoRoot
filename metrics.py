import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from skimage.measure import label, euler_number
from sklearn.metrics import precision_recall_fscore_support, f1_score

class RootMetrics:
    """
    Classe regroupant des métriques pour l'évaluation de segmentation de racines.
    Accepte des numpy arrays binaires (0 ou 1/255).
    """

    @staticmethod
    def _prepare(mask):
        """Convertit en binaire booléen (0/1)."""
        return (mask > 0).astype(bool)

    @staticmethod
    def get_pixel_metrics(y_true, y_pred):
        """
        Retourne Dice, Precision, Recall, F1-Score au niveau pixel.
        Note: Pour la segmentation binaire, F1-Score est équivalent au Dice.
        """
        y_true = RootMetrics._prepare(y_true).flatten()
        y_pred = RootMetrics._prepare(y_pred).flatten()

        # average='binary' considère la classe 1 comme la classe positive
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        # Dice est mathématiquement équivalent au F1 pour le binaire
        dice = f1 
        
        return {
            "dice": dice,
            "f1_score": f1,
            "precision": precision,
            "recall": recall
        }

    @staticmethod
    def get_hausdorff_distance(y_true, y_pred):
        """
        Calcule la distance de Hausdorff (95% percentile est souvent préféré, 
        ici implémentation standard max(min)).
        Utilise scipy pour l'efficacité.
        """
        y_true = RootMetrics._prepare(y_true)
        y_pred = RootMetrics._prepare(y_pred)

        # Si l'une des images est vide
        if not np.any(y_true) or not np.any(y_pred):
            return np.inf

        # Récupération des coordonnées des points (pixels actifs)
        u_points = np.argwhere(y_true)
        v_points = np.argwhere(y_pred)

        # directed_hausdorff(u, v) calcule max(min(d(u, v)))
        # La distance de Hausdorff est le max des deux directions
        forward = directed_hausdorff(u_points, v_points)[0]
        backward = directed_hausdorff(v_points, u_points)[0]

        return max(forward, backward)

    @staticmethod
    def get_betti_numbers(mask):
        """
        Calcule les nombres de Betti (Topologie).
        Betti 0 : Nombre de composantes connexes (racines détachées).
        Betti 1 : Nombre de cycles/trous (boucles dans la racine).
        """
        mask = RootMetrics._prepare(mask)
        
        # Betti 0: Composantes connexes
        # connectivity=2 permet les diagonales (8-voisins)
        labeled_mask, num_components = label(mask, return_num=True, connectivity=2)
        betti_0 = num_components

        # Betti 1: Calculé via la caractéristique d'Euler (Chi)
        # Pour une image 2D : Chi = Betti_0 - Betti_1
        # Donc Betti_1 = Betti_0 - Chi
        euler = euler_number(mask, connectivity=2)
        betti_1 = betti_0 - euler

        return {"betti_0": betti_0, "betti_1": betti_1}

    @staticmethod
    def get_skeleton_metrics(y_true, y_pred):
        """
        Calcule les métriques basées sur le squelette (Topology-aware).
        1. clDice (Centerline Dice) : Robustesse aux variations d'épaisseur.
        2. Average Centerline Distance (ACD).
        """
        y_true = RootMetrics._prepare(y_true)
        y_pred = RootMetrics._prepare(y_pred)
        
        if not np.any(y_true) or not np.any(y_pred):
            return {"cl_dice": 0.0, "acd": np.inf}

        # Squelettisation
        skel_true = skeletonize(y_true)
        skel_pred = skeletonize(y_pred)

        # --- 1. clDice (Centerline Dice) ---
        # Tprec: Fraction du squelette prédit qui est dans le masque GT
        tprec = np.sum(skel_pred * y_true) / (np.sum(skel_pred) + 1e-6)
        
        # Tsens: Fraction du squelette GT qui est dans le masque prédit
        tsens = np.sum(skel_true * y_pred) / (np.sum(skel_true) + 1e-6)
        
        cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-6)

        # --- 2. Average Centerline Distance (ACD) ---
        # Distance de chaque pixel du squelette Pred vers le masque GT le plus proche
        # On utilise la Distance Transform inversée du GT
        dist_map_true = distance_transform_edt(np.logical_not(y_true))
        dist_map_pred = distance_transform_edt(np.logical_not(y_pred))
        
        # Distance moyenne du squelette Pred vers le masque GT
        acd_pred_to_gt = np.mean(dist_map_true[skel_pred]) if np.any(skel_pred) else 0
        
        # Distance moyenne du squelette GT vers le masque Pred
        acd_gt_to_pred = np.mean(dist_map_pred[skel_true]) if np.any(skel_true) else 0
        
        acd = (acd_pred_to_gt + acd_gt_to_pred) / 2.0

        return {"cl_dice": cl_dice, "acd": acd}

    @staticmethod
    def get_connectivity_metrics(mask):
        """
        Vérifie la connectivité du squelette.
        Retourne le ratio de la plus grande composante connexe sur la longueur totale.
        1.0 = Racine parfaitement connectée en un seul morceau.
        < 1.0 = Racine fragmentée.
        """
        mask = RootMetrics._prepare(mask)
        if not np.any(mask): return 0.0
        
        skel = skeletonize(mask)
        if not np.any(skel): return 0.0
        
        # Etiquetage des morceaux du squelette
        labels, num = label(skel, return_num=True, connectivity=2)
        
        if num == 0: return 0.0
        if num == 1: return 1.0 # Un seul morceau
        
        # Calcul de la taille de chaque composante
        counts = np.bincount(labels.ravel())
        # counts[0] est le fond, on l'ignore
        counts = counts[1:]
        
        max_component = counts.max()
        total_pixels = counts.sum()
        
        return max_component / total_pixels

    @staticmethod
    def compute_all(y_true, y_pred):
        """Wrapper global pour calculer toutes les métriques d'un coup."""
        
        # 1. Pixel-wise
        pixel = RootMetrics.get_pixel_metrics(y_true, y_pred)
        
        # 2. Distance
        hausdorff = RootMetrics.get_hausdorff_distance(y_true, y_pred)
        
        # 3. Topologie (Comparaison des nombres de Betti)
        betti_true = RootMetrics.get_betti_numbers(y_true)
        betti_pred = RootMetrics.get_betti_numbers(y_pred)
        betti_error_0 = abs(betti_true['betti_0'] - betti_pred['betti_0'])
        betti_error_1 = abs(betti_true['betti_1'] - betti_pred['betti_1'])
        
        # 4. Squelette (Centerline Dice & ACD)
        skeleton = RootMetrics.get_skeleton_metrics(y_true, y_pred)
        
        # 5. Connectivité (Fragmentation)
        connectivity_gt = RootMetrics.get_connectivity_metrics(y_true)
        connectivity_pred = RootMetrics.get_connectivity_metrics(y_pred)
        
        return {
            **pixel,
            "hausdorff_dist": hausdorff,
            "betti_0_error": betti_error_0,
            "betti_1_error": betti_error_1,
            "gt_betti_0": betti_true['betti_0'],
            "pred_betti_0": betti_pred['betti_0'],
            **skeleton,
            "connectivity_ratio_gt": connectivity_gt,
            "connectivity_ratio_pred": connectivity_pred
        }

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    # Simulation d'images (256x256)
    gt = np.zeros((256, 256), dtype=np.uint8)
    pred = np.zeros((256, 256), dtype=np.uint8)
    
    # On dessine une ligne (racine)
    gt[50:200, 100:105] = 1
    # La prédiction est un peu décalée et cassée au milieu (trou)
    pred[50:120, 102:107] = 1
    pred[130:200, 102:107] = 1
    
    results = RootMetrics.compute_all(gt, pred)
    
    print("=== Résultats des métriques ===")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    
    print("\nInterprétation :")
    print(f"Différence de connectivité : Le GT est à {results['connectivity_ratio_gt']*100}% connecté, "
          f"la Préd est à {results['connectivity_ratio_pred']*100}% (donc cassée).")