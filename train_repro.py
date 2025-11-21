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

# --- Configuration ---
CONF = {
    'tileSize': [256, 256],
    'batchSize': 8,
    'numEpochs': 100,
    'iterPerEpoch': 10,  #  1000,
    'learning_rate': 0.0001,
    'dropout': 0.30,
    'loss': 'cross_entropy',
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

def evaluate_validation(sess, net, data_val, gt_val, conf, writer, epoch):
    """
    Évalue le modèle avec des métriques avancées sur l'ensemble de validation.
    """
    # Initialisation des accumulateurs
    metrics_sum = {
        'loss': [], 'f1': [], 'precision': [], 'recall': [],
        'iou': [], 'dice': [], 'hausdorff': [],
        'betti_0_err': [], 'betti_1_err': [],
        'centerline_dist': []
    }
    
    keep_pred = None
    keep_gt = None

    for img, gt in tqdm.tqdm(zip(data_val, gt_val), total=len(data_val), desc="Validation (Detailed)", leave=False):
        
        # Préparation input
        img_input = (img.astype(np.float32) / 255.0)[np.newaxis, :, :, np.newaxis]
        gt_input = gt[np.newaxis, :, :, :] # (1, H, W, 2)

        # 1. Prédiction (Output réseau souvent softmax ou logits)
        # Assumons que net.segment renvoie les probabilités (1, H, W, 2)
        pred_prob = net.segment(img_input) 
        
        # Calcul de la loss (approximatif si on n'a pas accès au graphe interne ici, sinon on ignore)
        # Si net.deploy renvoie la loss, utilisez net.deploy. Ici on met 0 ou on calcule manuellement.
        current_loss = 0.0 # Placeholder si non disponible via .segment()

        # 2. Post-Processing : Binarisation
        # On prend le canal 1 (Foreground)
        foreground_prob = pred_prob[0, :, :, 1]
        pred_mask = (foreground_prob > conf.get('Thresh', 0.5)).astype(np.uint8)
        
        # GT Mask (Canal 1 est le foreground dans votre load_folder)
        gt_mask = gt[:, :, 1].astype(np.uint8)

        keep_pred = pred_mask
        keep_gt = gt_mask
        # 3. Calcul des métriques avancées
        results = compute_advanced_metrics(pred_mask, gt_mask)
        
        # Stockage
        metrics_sum['loss'].append(current_loss)
        for k, v in results.items():
            if k in metrics_sum:
                metrics_sum[k].append(v)

    # Moyenne sur tout le dataset de validation
    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items()}

    # Logging TensorBoard
    print(f"\n--- Epoch {epoch} Validation Results ---")
    for k, v in avg_metrics.items():
        writer.add_summary(make_summary(f'val_{k}', v), epoch)
        print(f"  {k}: {v:.4f}")

    if keep_pred is not None and keep_gt is not None:
        cv2.imwrite(os.path.join(conf['logDirRoot'], f'val_pred_epoch_{epoch}.png'), keep_pred * 255)
        cv2.imwrite(os.path.join(conf['logDirRoot'], f'val_gt_epoch_{epoch}.png'), keep_gt * 255)
    return avg_metrics

def train_one_model(model_name, d_train, g_train, d_val, g_val, input_dir):
    print(f"=== Entraînement : {model_name} ===")
    tf.compat.v1.reset_default_graph()

    current_conf = CONF.copy()
    current_conf['Model'] = model_name
    current_conf['l2'] = MODEL_L2.get(model_name, 1e-9)

    ckpt_base_path = os.path.join(current_conf['ckptDirRoot'], model_name)

    log_path_train = os.path.join(
        current_conf['logDirRoot'], model_name, 'train')
    log_path_val = os.path.join(current_conf['logDirRoot'], model_name, 'val')

    os.makedirs(ckpt_base_path, exist_ok=True)

    batch_gen = Patch2DBatchGeneratorFromTensors(
        current_conf, d_train, g_train, augment=True, infiniteLoop=True
    )
    batch_gen.generateBatches()

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config_proto) as sess:
        net = RootNet(sess, current_conf, model_name, isTrain=True)

        train_writer = tf.compat.v1.summary.FileWriter(
            log_path_train, sess.graph)
        val_writer = tf.compat.v1.summary.FileWriter(log_path_val)

        global_step = 0
        epoch_pbar = tqdm.tqdm(
            range(current_conf['numEpochs']), desc="Epochs", unit="ep")

        for epoch in epoch_pbar:
            epoch_loss = 0
            batch_pbar = tqdm.tqdm(range(
                current_conf['iterPerEpoch']), desc=f"Epoch {epoch+1}", leave=False, unit="batch")

            for _ in batch_pbar:
                try:
                    batch_x, batch_y = batch_gen.queue.get(timeout=60)
                    batch_gen.queue.task_done()
                except queue.Empty:
                    print("Erreur: Timeout lors de la récupération du batch.")
                    break

                loss = net.fit(
                    batch_x, batch_y, learning_rate=current_conf['learning_rate'], phase=True)
                epoch_loss += loss

                train_writer.add_summary(make_summary(
                    'batch_loss', loss), global_step)
                global_step += 1
                batch_pbar.set_postfix({'loss': f"{loss:.4f}"})

            avg_train_loss = epoch_loss / current_conf['iterPerEpoch']
            train_writer.add_summary(make_summary(
                'epoch_loss', avg_train_loss), epoch)

            val_msg = ""
            if len(d_val) > 0:
                print("  -> Validation en cours (Tentative Full-Res)...")
                metrics = evaluate_validation(
                    sess, net, d_val, g_val, current_conf, val_writer, epoch)
                print(f"  -> Val Dice: {metrics['dice']:.4f}")

                # Sauvegarde Checkpoint
                epoch_dir = os.path.join(ckpt_base_path, f"epoch_{epoch+1}")
                os.makedirs(epoch_dir, exist_ok=True)
                net.save(epoch_dir)

            epoch_pbar.set_postfix({'Train Loss': f"{avg_train_loss:.4f}"})

            epoch_save_dir = os.path.join(ckpt_base_path, f"epoch_{epoch + 1}")

            os.makedirs(epoch_save_dir, exist_ok=True)

            print(f" -> Sauvegarde modèle dans : {epoch_save_dir}")
            net.save(epoch_save_dir)

            train_writer.flush()
            val_writer.flush()

    batch_gen.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default="./Data")
    parser.add_argument('--models', type=str, nargs='+',
                        default=['UNet', 'ResUNet', 'ResUNetDS', 'SegNet', 'DeepLab'])
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
                            d_val, g_val, args.input_dir)
        except Exception as e:
            print(f"Erreur sur {model}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
