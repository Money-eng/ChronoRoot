import tensorflow as tf
import tqdm
import os
import numpy as np
import argparse
import cv2
import queue
from rootNet.Model import RootNet
from rootNet.BatchGenerator import Patch2DBatchGeneratorFromTensors
from rootNet.Provider import MPImageDataProvider

# --- Configuration ---
CONF = {
    'tileSize': [256, 256],
    'batchSize': 8,
    'numEpochs': 50,
    'iterPerEpoch': 41, # number of images of train // batchSize
    'learning_rate': 0.0001,
    'dropout': 0.30,
    'loss': 'cross_entropy',
    'lambda1': 0.5,
    'lambda2': 0.5,
    'ckptDirRoot': 'modelWeights',
    'multipleOf': [32, 32],
    'OriginalSize': [2464, 3280],
    'Alpha': 0.8,  # Set to 0 if no postprocess wanted
    'Thresh': 0.5,
    'PostProcess': True,
    'timeStep': 15,  # In minutes
    'SmoothFactor': 8,
    'logDirRoot': 'logs'  # Nouveau dossier pour TensorBoard
}

MODEL_L2 = {
    'UNet': 1e-8, 'ResUNet': 1e-8, 'ResUNetDS': 1e-8, 'DeepLab': 1e-9, 'SegNet': 1e-10
}


def load_folder(folder_path, img_suffix, mask_suffix, desc):
    """Charge un dossier spécifique (Train ou Test)."""
    if not os.path.exists(folder_path):
        print(f"ATTENTION: Le dossier {folder_path} n'existe pas !")
        return [], []

    # Désactive l'augmentation et le shuffle ici, c'est géré par le BatchGenerator plus tard
    provider = MPImageDataProvider(search_path=[
                                   folder_path], data_suffix=img_suffix, mask_suffix=mask_suffix, augment=False, shuffle_data=False)

    data = []
    gt = []

    # Utilisation de tqdm pour voir l'avancement du chargement
    for img_path in tqdm.tqdm(provider.data_files, desc=desc):
        img = cv2.imread(img_path, 0)  # 2D (H, W)

        mask_path = img_path.replace(img_suffix, mask_suffix)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            continue

        # Image : On garde en 2D (H, W) -> Le BatchGenerator gère l'ajout de dimension
        data.append(img)

        # Masque : On convertit en One-Hot 3D (H, W, 2)
        mask_bool = mask > 0
        mask_cat = np.zeros(
            (mask.shape[0], mask.shape[1], 2), dtype=np.float32)
        mask_cat[:, :, 0] = np.logical_not(mask_bool)  # Background
        mask_cat[:, :, 1] = mask_bool                 # Foreground
        gt.append(mask_cat)

    return data, gt


def load_dataset(input_dir, img_suffix=".png", mask_suffix="_mask.png"):
    """Charge les données depuis les sous-dossiers Train et Test."""
    print(f"Chargement des données depuis {input_dir}...")

    # On cherche explicitement les dossiers Train et Test (ou Validation)
    # Adaptez les noms si vos dossiers s'appellent autrement (ex: 'val', 'validation')
    train_dir = os.path.join(input_dir, 'Train')
    test_dir = os.path.join(input_dir, 'Test')

    # Si 'Test' n'existe pas, essayer 'Validation' ou 'val'
    if not os.path.exists(test_dir):
        if os.path.exists(os.path.join(input_dir, 'Validation')):
            test_dir = os.path.join(input_dir, 'Validation')
        elif os.path.exists(os.path.join(input_dir, 'val')):
            test_dir = os.path.join(input_dir, 'val')

    data_train, gt_train = load_folder(
        train_dir, img_suffix, mask_suffix, "Chargement Train")
    data_val, gt_val = load_folder(
        test_dir, img_suffix, mask_suffix, "Chargement Test")

    print(
        f"Résumé : {len(data_train)} images d'entraînement, {len(data_val)} images de validation.")
    return data_train, gt_train, data_val, gt_val


def make_summary(name, value):
    """Crée un objet Summary manuellement pour logger une valeur scalaire dans TF1."""
    return tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=name, simple_value=float(value))])

def compute_numpy_metrics(pred_prob, gt_mask, threshold=0.5):
    """
    Calcule Dice, Precision, Recall sur des tableaux Numpy (H, W).
    pred_prob: Probabilité (0.0 à 1.0)
    gt_mask: Vérité terrain (0 ou 1)
    """
    # Binarisation
    pred_mask = (pred_prob > threshold).astype(np.uint8)
    gt_mask = (gt_mask > 0).astype(np.uint8)
    
    # Intersection et Unions
    # On travaille sur des booléens pour la vitesse
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()
    
    # Dice = 2*Inter / (Pred + GT)
    smooth = 1e-6 # Pour éviter division par zéro
    dice = (2. * intersection + smooth) / (pred_sum + gt_sum + smooth)
    
    # Precision = TP / (TP + FP) = Inter / Pred
    precision = (intersection + smooth) / (pred_sum + smooth)
    
    # Recall = TP / (TP + FN) = Inter / GT
    recall = (intersection + smooth) / (gt_sum + smooth)
    
    return dice, precision, recall

def evaluate_validation(sess, net, data_val, gt_val, conf, writer, epoch):
    """
    1. Reconstruit les images complètes par tuilage.
    2. Calcule les métriques réelles sur l'image entière via Numpy.
    3. Envoie la dernière image traitée à TensorBoard.
    """
    total_dice, total_prec, total_rec = 0, 0, 0
    
    # Limite pour ne pas y passer 1h si le val set est énorme (ex: 20 images max)
    # Si votre val set est petit (<50 images), vous pouvez enlever cette limite [:20]
    val_subset_img = data_val[:20] 
    val_subset_gt = gt_val[:20]
    n_images = len(val_subset_img)
    
    last_vis_img = None
    last_vis_str = None

    # Paramètres de tuilage
    tile_h, tile_w = conf['tileSize']
    batch_size = conf['batchSize']
    
    for i in range(n_images):
        full_img = val_subset_img[i] # (H, W)
        full_gt = val_subset_gt[i]   # (H, W, 2)
        
        h, w = full_img.shape
        full_pred = np.zeros((h, w), dtype=np.float32)
        
        # Padding
        pad_h = (tile_h - h % tile_h) % tile_h
        pad_w = (tile_w - w % tile_w) % tile_w
        img_padded = np.pad(full_img, ((0, pad_h), (0, pad_w)), mode='reflect')
        
        # --- Tuilage et Prédiction ---
        for y in range(0, img_padded.shape[0], tile_h):
            for x in range(0, img_padded.shape[1], tile_w):
                tile = img_padded[y:y+tile_h, x:x+tile_w]
                
                # Dummy Batch pour tromper le réseau statique
                dummy_batch = np.zeros((batch_size, tile_h, tile_w, 1), dtype=np.float32)
                dummy_batch[0, :, :, 0] = tile
                
                batch_seg = net.segment(dummy_batch)
                tile_prob = batch_seg[0, :, :, 1]
                
                # Reconstruction
                h_end = min(y+tile_h, h)
                w_end = min(x+tile_w, w)
                valid_h = h_end - y
                valid_w = w_end - x
                full_pred[y:h_end, x:w_end] = tile_prob[:valid_h, :valid_w]
        
        # --- Calcul Métriques Numpy (Sur l'image complète) ---
        # On compare le canal 1 du GT (Racine) avec la prédiction
        dc, prec, rec = compute_numpy_metrics(full_pred, full_gt[:, :, 1])
        
        total_dice += dc
        total_prec += prec
        total_rec += rec
        
        # --- Préparation Image TensorBoard (Uniquement pour la dernière) ---
        if i == n_images - 1:
            vis_img = cv2.cvtColor(full_img, cv2.COLOR_GRAY2BGR)
            
            # Vert = GT
            gt_mask = (full_gt[:, :, 1] > 0).astype(np.uint8)
            contours_gt, _ = cv2.findContours(gt_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours_gt, -1, (0, 255, 0), 2)
            
            # Rouge = Pred
            pred_mask = (full_pred > 0.5).astype(np.uint8)
            contours_pred, _ = cv2.findContours(pred_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis_img, contours_pred, -1, (0, 0, 255), 1)
            
            # Resize pour logs légers
            scale = 25
            dim = (int(vis_img.shape[1]*scale/100), int(vis_img.shape[0]*scale/100))
            vis_resized = cv2.resize(vis_img, dim, interpolation=cv2.INTER_AREA)
            _, buf = cv2.imencode('.png', vis_resized)
            last_vis_str = buf.tobytes()
            last_vis_shape = dim

    # --- Envoi TensorBoard ---
    if writer is not None and last_vis_str is not None:
        summary = tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(tag="Validation_Full_Reconstruction", 
                                       image=tf.compat.v1.Summary.Image(encoded_image_string=last_vis_str,
                                                                        height=last_vis_shape[1], 
                                                                        width=last_vis_shape[0]))
        ])
        writer.add_summary(summary, epoch)

    # Moyennes
    avg_dice = total_dice / n_images
    avg_prec = total_prec / n_images
    avg_rec = total_rec / n_images
    
    # Note: La "loss" est difficile à calculer exactement sur l'image entière sans TF
    # On peut retourner 0 ou une approximation, le Dice est le plus important ici.
    return {
        'loss': 1.0 - avg_dice, 
        'dice': avg_dice,
        'precision': avg_prec,
        'recall': avg_rec
    }

def train_one_model(model_name, d_train, g_train, d_val, g_val, input_dir):
    print(f"=== Entraînement : {model_name} ===")
    tf.compat.v1.reset_default_graph()
    
    current_conf = CONF.copy()
    current_conf['Model'] = model_name
    current_conf['l2'] = MODEL_L2.get(model_name, 1e-9)
    
    ckpt_path = os.path.join(current_conf['ckptDirRoot'], model_name)
    log_path_train = os.path.join(current_conf['logDirRoot'], model_name, 'train')
    log_path_val = os.path.join(current_conf['logDirRoot'], model_name, 'val')
    os.makedirs(ckpt_path, exist_ok=True)

    batch_gen = Patch2DBatchGeneratorFromTensors(
        current_conf, d_train, g_train, augment=True, infiniteLoop=True
    )
    batch_gen.generateBatches()
    
    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    
    with tf.compat.v1.Session(config=config_proto) as sess:
        net = RootNet(sess, current_conf, model_name, isTrain=True)
        
        train_writer = tf.compat.v1.summary.FileWriter(log_path_train, sess.graph)
        val_writer = tf.compat.v1.summary.FileWriter(log_path_val)
        
        global_step = 0
        epoch_pbar = tqdm.tqdm(range(current_conf['numEpochs']), desc="Epochs", unit="ep")
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            batch_pbar = tqdm.tqdm(range(current_conf['iterPerEpoch']), desc=f"Ep {epoch+1}", leave=False, unit="batch")
            
            for i in batch_pbar:
                try:
                    batch_x, batch_y = batch_gen.queue.get(timeout=30)
                    batch_gen.queue.task_done()
                except queue.Empty:
                    print("\n[ERREUR] Timeout générateur !")
                    batch_gen.finish()
                    return

                loss = net.fit(batch_x, batch_y, learning_rate=current_conf['learning_rate'], phase=True)
                epoch_loss += loss
                
                train_writer.add_summary(make_summary('batch_loss', loss), global_step)
                global_step += 1
                batch_pbar.set_postfix({'loss': f"{loss:.4f}"})
            
            avg_train_loss = epoch_loss / current_conf['iterPerEpoch']
            train_writer.add_summary(make_summary('epoch_loss', avg_train_loss), epoch)
            
            val_msg = ""
            if len(d_val) > 0:
                # On passe le writer et l'époque à la fonction
                metrics = evaluate_validation(sess, net, d_val, g_val, current_conf, val_writer, epoch)
                
                val_writer.add_summary(make_summary('epoch_loss', metrics['loss']), epoch)
                val_writer.add_summary(make_summary('dice_score', metrics['dice']), epoch)
                val_writer.add_summary(make_summary('precision', metrics['precision']), epoch)
                val_writer.add_summary(make_summary('recall', metrics['recall']), epoch)
                
                val_msg = f" | Val Dice: {metrics['dice']:.4f}"

            epoch_pbar.set_postfix({'Train Loss': f"{avg_train_loss:.4f}", 'Val': val_msg})
            
            net.save(ckpt_path)
            train_writer.flush()
            val_writer.flush()

    batch_gen.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default="/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/Data")
    parser.add_argument('--models', type=str, nargs='+',
                        default=['UNet', 'ResUNet', 'ResUNetDS', 'SegNet', 'DeepLab'])
    args = parser.parse_args()

    d_train, g_train, d_val, g_val = load_dataset(args.input_dir)

    if len(d_train) == 0:
        print("Erreur: Pas de données.")
        return

    for model in args.models:
        try:
            train_one_model(model, d_train, g_train,
                            d_val, g_val, args.input_dir)
        except Exception as e:
            print(f"Erreur sur {model}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
