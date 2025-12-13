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
from metrics import compute_advanced_metrics
from skimage.morphology import remove_small_objects
import pydensecrf.densecrf as dcrf
import concurrent.futures

# --- Configuration ---
CONF = {
    'tileSize': [256, 256],
    'batchSize': 8,
    'numEpochs': 200,
    'iterPerEpoch': 100,
    'learning_rate': 0.005,
    'dropout': 0.30,
    'loss': 'dice',
    'lambda1': 0.5,
    'lambda2': 0.5,
    'ckptDirRoot': 'modelWeights',
    'multipleOf': [32, 32],
    'OriginalSize': [2464, 3280],
    'Alpha': 0.9,  # Set to 0 if no postprocess wanted
    'Thresh': 0.5,
    'timeStep': 15,
    'PostProcess': True,
    'SmoothFactor': 8,
    'logDirRoot': 'logs'  # Nouveau dossier pour TensorBoard
}

MODEL_L2 = {
    'UNet': 1e-8, 'ResUNet': 1e-8, 'ResUNetDS': 1e-8, 'DeepLab': 1e-9, 'SegNet': 1e-10
}

HEAVY_METRICS_FREQ = 5

def load_folder(folder_path, img_suffix, mask_suffix, desc):
    if not os.path.exists(folder_path):
        print(f"ATTENTION: Le dossier {folder_path} n'existe pas !")
        return [], []

    provider = MPImageDataProvider(search_path=[
                                   folder_path], data_suffix=img_suffix, mask_suffix=mask_suffix, augment=False, shuffle_data=False)

    data = []
    gt = []

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
    print(f"Chargement des données depuis {input_dir}...")

    train_dir = os.path.join(input_dir, 'Train')
    test_dir = os.path.join(input_dir, 'Test')

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


def evaluate_validationOLD(sess, net, data_val, gt_val, conf, writer, epoch):
    """
    Évalue le modèle sur l'ensemble de validation complet (Full Resolution).
    """
    losses = []
    dices = []

    for img, gt in tqdm.tqdm(zip(data_val, gt_val), total=len(data_val), desc="Validation", leave=False):

        img_input = (img.astype(np.float32) /
                     255.0)[np.newaxis, :, :, np.newaxis]

        gt_input = gt[np.newaxis, :, :, :]

        loss, dice, auc, prec, rec = net.deploy(img_input, gt_input, phase=0)

        losses.append(loss)
        dices.append(dice)

    avg_loss = np.mean(losses)
    avg_dice = np.mean(dices)

    writer.add_summary(make_summary('val_loss', avg_loss), epoch)
    writer.add_summary(make_summary('val_dice', avg_dice), epoch)

    return {'loss': avg_loss, 'dice': avg_dice}


def evaluate_validation(sess, net, data_val, gt_val, conf, writer, epoch, model_name, do_heavy, use_crf=True):
    """
    Version parallélisée de l'évaluation.
    """
    metrics_sum = {
        'loss': [], 'f1': [], 'precision': [], 'recall': [], 'iou': [], 'dice': [], 
        'dice_gpu': [], 'auc_gpu': [], 'precision_gpu': [], 'recall_gpu': [], 'betti_0_err': [], 'betti_1_err': [], 'betti_0_abs_err': [], 'betti_1_abs_err': [], 'centerline_distance': [], 'hausdorff_95': [], 'hausdorff_max': [], 'surface_dice_1mm': []
    }
    
    # On ajoute les clés lourdes seulement si nécessaire pour ne pas polluer les logs avec des 0
    if do_heavy:
        heavy_keys = ['apls', 'apls_recall', 'apls_precision']
        for k in heavy_keys:
            metrics_sum[k] = []
    
    # using deploy to compute loss only
    loss, diceg, aucg, precg, recg = net.deploy(data_val[0][np.newaxis, :, :, np.newaxis],
                         gt_val[0][np.newaxis, :, :, :])
    metrics_sum['loss'].append(loss)
    metrics_sum['dice_gpu'].append(diceg)
    metrics_sum['auc_gpu'].append(aucg)
    metrics_sum['precision_gpu'].append(precg)
    metrics_sum['recall_gpu'].append(recg)
    

    tasks = []

    do_heavy_img = do_heavy
    for img, gt in zip(data_val, gt_val): # img is shape (H, W), gt is shape (H, W, 2)
        img_input = (img.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis] # shape (1, H, W, 1)

        pred_prob = net.segment(img_input)
            
        tasks.append((pred_prob.copy(), img.copy(), gt.copy(), conf, use_crf, do_heavy))

    keep_pred = None
    keep_gt = None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_list = list(tqdm.tqdm(executor.map(
            process_single_validation_item, tasks), total=len(tasks), desc=f"Valid (Heavy={do_heavy_img})"))

    for i, (res, p_mask, g_mask) in enumerate(results_list):
        if i == len(results_list) - 1:
            keep_pred = p_mask
            keep_gt = g_mask

        metrics_sum['loss'].append(0.0)

        for k, v in res.items():
            if k in metrics_sum:
                metrics_sum[k].append(v)

    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items() if v}

    print(f"\n--- Epoch {epoch} Validation Results ---")
    for k, v in avg_metrics.items():
        writer.add_summary(make_summary(f'val_{k}', v), epoch)
        print(f"  {k}: {v:.4f}")

    if keep_pred is not None and keep_gt is not None:
        save_dir = os.path.join(conf['logDirRoot'], f"model_{model_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(
            save_dir, f'val_pred_epoch_{epoch}.png'), keep_pred * 255)
        cv2.imwrite(os.path.join(
            save_dir, f'val_gt_epoch_{epoch}.png'), keep_gt * 255)

    return avg_metrics


def process_single_validation_item(args):
    """
    Fonction worker qui s'exécute sur un processeur séparé.
    Elle reçoit les données brutes (numpy) et renvoie les métriques.
    """
    pred_prob, img, gt, conf, use_crf, do_heavy = args

    # 1. Post-traitement (CRF ou Seuil simple) - COPIÉ DE VOTRE CODE
    if False:
        image_rgb = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        # Ajustement des dimensions pour dcrf
        label_1 = np.transpose(pred_prob[0, :, :, :], (2, 0, 1))
        unary = -np.log(np.clip(label_1, 1e-5, 1.0))
        _, H, W = unary.shape
        unary = unary.transpose(0, 2, 1)
        unary = unary.reshape(2, -1)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image_rgb, compat=1)
        Q = d.inference(1)
        crf_map = np.array(Q).reshape(2, H, W).transpose(1, 2, 0)
        prob_map = crf_map[:, :, 1]
    else:
        prob_map = pred_prob[0, :, :, 1]

    pred_mask_bin = (prob_map > conf.get('Thresh', 0.5)).astype(np.uint8).astype(bool)

    if True:
        min_size = 25
        pred_mask = remove_small_objects(
            pred_mask_bin, min_size=min_size).astype(np.uint8)
    else:
        pred_mask = pred_mask_bin.astype(np.uint8)

    gt_mask = gt[:, :, 1].astype(np.uint8)

    # 2. Calcul des métriques lourdes
    results = compute_advanced_metrics(pred_mask, gt_mask, do_heavy)

    # On retourne aussi les masques si c'est la dernière image (pour la sauvegarde)
    return results, pred_mask, gt_mask


def train_one_model(model_name, d_train, g_train, d_val, g_val):
    print(f"=== Entraînement : {model_name} ===")
    tf.compat.v1.reset_default_graph()

    current_conf = CONF.copy()
    current_conf['Model'] = model_name
    current_conf['l2'] = MODEL_L2.get(model_name, 1e-9)

    # --- CONFIGURATION DU SCHEDULER ---
    current_lr = current_conf['learning_rate'] # LR dynamique
    lr_patience = 10        # Nombre d'époques à attendre avant de réduire
    lr_factor = 0.5         # Facteur de réduction (ex: on divise par 2)
    lr_min = 1e-6           # Ne pas descendre en dessous de ça
    lr_wait = 0             # Compteur d'attente
    best_val_dice = float('-inf') # Meilleure loss vue jusqu'ici
    # ----------------------------------

    ckpt_base_path = os.path.join(current_conf['ckptDirRoot'], model_name)
    log_path_train = os.path.join(current_conf['logDirRoot'], model_name, 'train')
    log_path_val = os.path.join(current_conf['logDirRoot'], model_name, 'val')

    os.makedirs(ckpt_base_path, exist_ok=True)

    batch_gen = Patch2DBatchGeneratorFromTensors(
        current_conf, d_train, g_train, augment=True, infiniteLoop=True, maxQueueSize=200
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
            # ... (Code existant de chargement de checkpoint inchangé) ...
            epoch_save_dir = os.path.join(ckpt_base_path, f"epoch_{epoch + 1}")
            checkpoint_exists = False
            if os.path.exists(epoch_save_dir):
                if os.path.exists(os.path.join(epoch_save_dir, "checkpoint")):
                    checkpoint_exists = True

            if checkpoint_exists:
                tqdm.tqdm.write(f" -> Checkpoint for epoch {epoch+1} already exists. Loading model...")
                net.restore(epoch_save_dir)

            epoch_loss = 0.0
            batch_pbar = tqdm.tqdm(range(current_conf['iterPerEpoch']), desc=f"Epoch {epoch+1}", leave=False)
            
            for _ in batch_pbar:
                try:
                    batch_x, batch_y = batch_gen.queue.get(timeout=60)
                    batch_gen.queue.task_done()
                except queue.Empty:
                    print("Erreur: Timeout lors de la récupération du batch.")
                    break

                if checkpoint_exists:
                    loss, _, _, _, _ = net.deploy(batch_x, batch_y, phase=False)
                    epoch_loss += loss
                    train_writer.add_summary(make_summary('batch_loss', loss), global_step)
                    global_step += 1
                    batch_pbar.set_postfix({'loss': f"{loss:.4f}"})
                else:
                    # MODIFICATION ICI : Utiliser current_lr au lieu de la config statique
                    loss = net.fit(batch_x, batch_y, learning_rate=current_lr, phase=True)
                    
                    epoch_loss += loss
                    train_writer.add_summary(make_summary('batch_loss', loss), global_step)
                    global_step += 1
                    
                    # On affiche le LR actuel dans la barre de progression pour suivi
                    batch_pbar.set_postfix({'loss': f"{loss:.4f}", 'lr': f"{current_lr:.1e}"})

            avg_train_loss = epoch_loss / current_conf['iterPerEpoch']
            train_writer.add_summary(make_summary('epoch_loss', avg_train_loss), epoch)
            # Logger le LR pour le suivre dans TensorBoard
            train_writer.add_summary(make_summary('learning_rate', current_lr), epoch)

            epoch_dir = os.path.join(ckpt_base_path, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)
            
            if len(d_val) > 0:
                do_heavy = ((epoch + 1) % HEAVY_METRICS_FREQ == 0) or ((epoch + 1) == current_conf['numEpochs'])
                
                # Evaluation
                metrics = evaluate_validation(sess, net, d_val, g_val, current_conf, val_writer, epoch, model_name, do_heavy)
                print(f"Metriques de validation à la fin de l'époque {epoch+1} : {metrics}")
                
                # --- LOGIQUE DU SCHEDULER ICI ---
                if not checkpoint_exists: # Ne pas changer le LR si on vient de recharger un vieux checkpoint
                    val_dice = metrics['dice'] # Ou metrics['dice'] si vous voulez maximiser le Dice (inverser la logique)
                    
                    # On cherche à maximiser le Dice (delta de 1e-4 pour éviter le bruit)
                    if val_dice > (best_val_dice + 1e-4):
                        best_val_dice = val_dice
                        lr_wait = 0 # On reset le compteur car on s'est amélioré
                    else:
                        lr_wait += 1
                        print(f" -> Validation loss ne s'améliore pas (Patience: {lr_wait}/{lr_patience})")
                        
                        if lr_wait >= lr_patience:
                            old_lr = current_lr
                            current_lr = max(current_lr * lr_factor, lr_min)
                            lr_wait = 0 # Reset patience après réduction
                            if current_lr < old_lr:
                                print(f"⚠️ PLATEAU DÉTECTÉ : Réduction du Learning Rate de {old_lr:.1e} à {current_lr:.1e}")
                # --------------------------------

            epoch_pbar.set_postfix({'Train Loss': f"{avg_train_loss:.4f}", 'Val Loss': f"{metrics.get('loss', 0):.4f}"})

            print(f" -> Sauvegarde modèle dans : {epoch_dir}")
            net.save(epoch_dir)
            train_writer.flush()
            val_writer.flush()

    batch_gen.finish()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default="./Data")
    parser.add_argument('--models', type=str, nargs='+',
                        default=['ResUNetDS'])
    args = parser.parse_args()

    print(f"Démarrage de l'entraînement pour les modèles : {args.models}")

    d_train, g_train, d_val, g_val = load_dataset(args.input_dir)

    if len(d_train) == 0:
        print("Erreur: Pas de données.")
        return

    for model in args.models:
        print(f"\n\n=== Entraînement du modèle : {model} ===")
        try:
            train_one_model(model, d_train, g_train,
                            d_val, g_val)
        except Exception as e:
            print(f"Erreur sur {model}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
