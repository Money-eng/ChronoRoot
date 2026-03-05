import tensorflow as tf
import wandb
import tqdm
import os
import numpy as np
import argparse
import cv2
from rootNet.Model import RootNet
from rootNet.Provider import MPImageDataProvider
from metrics import compute_advanced_metrics
import pydensecrf.densecrf as dcrf
from concurrent.futures import ProcessPoolExecutor, as_completed

MODEL_L2 = {
    'UNet': 1e-8, 'ResUNet': 1e-8, 'ResUNetDS': 1e-8, 'DeepLab': 1e-9, 'SegNet': 1e-10
}

CONF = {
    'tileSize': [256, 256],
    'batchSize': 8,
    'numEpochs': 200,
    'iterPerEpoch': 100,
    'learning_rate': 0.0001,
    'dropout': 0.30,
    'loss': 'cldice',
    'lambda1': 0.5,
    'lambda2': 0.5,
    'ckptDirRoot': 'modelWeights',
    'multipleOf': [32, 32],
    'OriginalSize': [2464, 3280],
    'Alpha': 0.9,
    'Thresh': 0.5,
    'timeStep': 15,
    'PostProcess': True,
    'SmoothFactor': 8,
    'logDirRoot': 'logs'
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

        data.append(img)

        mask_bool = mask > 0
        mask_cat = np.zeros(
            (mask.shape[0], mask.shape[1], 2), dtype=np.float32)
        mask_cat[:, :, 0] = np.logical_not(mask_bool)
        mask_cat[:, :, 1] = mask_bool
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


def evaluate_validation(sess, net, data_val, gt_val, conf, writer, epoch, model_name, do_heavy, use_crf=True):
    metrics_sum = {
        'loss': [], 'f1_score': [], 'f_2_score': [], 'f_3_score': [], 'f_4_score': [], 'precision': [], 'recall': [], 'specificity': [], 'mean_iou': [], 'iou': [], 'pixel_accuracy': [], 'dice': [], 'dice_gpu': [], 'auc_gpu': [], 'precision_gpu': [], 'recall_gpu': [], "hausdorff_95": [], "hausdorff_max": [], "surface_dice_1mm": [], "ASCD": [], "betti_0_abs_err": [], "betti_1_abs_err": [], "betti0_jaccard_ratio": [], "betti1_jaccard_ratio": [], "betti0_relative_error": [], "betti1_relative_error": [], "betti0_variation_index": [], "betti1_variation_index": [], "b0_pred": [], "b1_pred": [], 'normalized_mutual_information': []
    }

    if do_heavy:
        heavy_keys = ['apls', 'apls_recall', 'apls_precision']
        for k in heavy_keys:
            metrics_sum[k] = []

    normalized_data_val = [(img.astype(np.float32) / 255.0)[:, :, np.newaxis] for img in data_val]
    futures_list = []

    print(f"--- Evaluation Epoch {epoch} (Heavy={do_heavy}) ---")

    MULTIPLE_OF = 32

    with ProcessPoolExecutor() as executor:
        for _, (img, gt) in tqdm.tqdm(enumerate(zip(normalized_data_val, gt_val)), total=len(data_val), desc="GPU Inference"):
            
            # Gestion du padding spécifique à SegNet
            h, w = img.shape[0], img.shape[1]
            pad_h = (MULTIPLE_OF - (h % MULTIPLE_OF)) % MULTIPLE_OF
            pad_w = (MULTIPLE_OF - (w % MULTIPLE_OF)) % MULTIPLE_OF

            if model_name == 'SegNet' and (pad_h > 0 or pad_w > 0):
                img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
                gt_padded = np.pad(gt, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            else:
                img_padded = img
                gt_padded = gt

            img_input = img_padded[np.newaxis, :, :, :]
            gt_input = gt_padded[np.newaxis, :, :, :]
            
            # Inférence GPU
            loss, diceg, aucg, precg, recg = net.deploy(img_input, gt_input, phase=0)
            metrics_sum['loss'].append(loss)
            metrics_sum['dice_gpu'].append(diceg)
            metrics_sum['auc_gpu'].append(aucg)
            metrics_sum['precision_gpu'].append(precg)
            metrics_sum['recall_gpu'].append(recg)

            pred_prob_padded = net.segment(img_input)

            # Recadrage post-prédiction si SegNet a été paddé
            if model_name == 'SegNet' and (pad_h > 0 or pad_w > 0):
                pred_prob = pred_prob_padded[:, :h, :w, :]
                img_crop = img_padded[:h, :w, :]
                gt_crop = gt_padded[:h, :w, :]
            else:
                pred_prob = pred_prob_padded
                img_crop = img_padded
                gt_crop = gt_padded

            future = executor.submit(process_single_validation_item,
                                     (pred_prob.copy(), img_crop.copy(), gt_crop.copy(), conf, use_crf, do_heavy))
            futures_list.append(future)

        for future in tqdm.tqdm(as_completed(futures_list), total=len(futures_list), desc="CPU Metrics"):
            try:
                res, _, _ = future.result()
                for k, v in res.items():
                    if k in metrics_sum:
                        metrics_sum[k].append(v)
            except Exception as e:
                print(f"Erreur dans un worker : {e}")

    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items() if v}

    print(f"\nRésultats Epoch {epoch}:")
    for k, v in avg_metrics.items():
        writer.add_summary(make_summary(f'val/{k}', v), epoch)
        print(f"  {k}: {v}")
        
    return avg_metrics

def process_single_validation_item(args):
    pred_prob, img, gt, conf, use_crf, do_heavy = args

    H, W = pred_prob[0].shape[:2]

    if use_crf:
        img_squeeze = np.squeeze(img)
        img_uint8 = (img_squeeze * 255.0).astype(np.uint8)
        image_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
        image_rgb = np.ascontiguousarray(image_rgb)

        unary = np.transpose(pred_prob[0], (2, 0, 1))
        unary = -np.log(np.clip(unary, 1e-5, 1.0))
        unary = unary.reshape(2, -1)
        unary = np.ascontiguousarray(unary)

        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(unary)
        d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image_rgb, compat=1)
        Q = d.inference(1)
        crf_map = np.array(Q).reshape(2, H, W)
        prob_map = crf_map[1, :, :]
    else:
        prob_map = pred_prob[0, :, :, 1]

    pred_mask = (prob_map > conf.get('Thresh', 0.5)).astype(np.uint8)
    gt_mask = gt[:, :, 1].astype(np.uint8)

    results = compute_advanced_metrics(pred_mask, gt_mask, do_heavy)

    return results, pred_mask, gt_mask

def check_reproducibility(net, data_val, model_name):
    print("\n" + "#"*60)
    print("Vérification de la reproductibilité (Run A vs Run B)")
    print("#"*60)
    
    sample_data = data_val
    normalized_data = [(img.astype(np.float32) / 255.0)[:, :, np.newaxis] for img in sample_data]
    
    MULTIPLE_OF = 32

    def predict_run(data):
        preds = []
        for img in data:
            h, w = img.shape[0], img.shape[1]
            pad_h = (MULTIPLE_OF - (h % MULTIPLE_OF)) % MULTIPLE_OF
            pad_w = (MULTIPLE_OF - (w % MULTIPLE_OF)) % MULTIPLE_OF
            
            if model_name == 'SegNet' and (pad_h > 0 or pad_w > 0):
                img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
            else:
                img_padded = img
                
            img_input = img_padded[np.newaxis, :, :, :]
            pred_prob_padded = net.segment(img_input)
            
            if model_name == 'SegNet' and (pad_h > 0 or pad_w > 0):
                preds.append(pred_prob_padded[:, :h, :w, :])
            else:
                preds.append(pred_prob_padded)
        return preds

    run_a_preds = predict_run(normalized_data)
        
    run_b_preds = predict_run(normalized_data)
        
    run_a_concat = np.concatenate(run_a_preds, axis=0)
    run_b_concat = np.concatenate(run_b_preds, axis=0)
    
    diff = np.abs(run_a_concat - run_b_concat)
    max_diff = diff.max()
    
    print(f"Différence maximale : {max_diff}")
    
    if max_diff > 1e-6:
        print("❌ ÉCHEC : Les prédictions ne sont pas déterministes.")
        return False
    else:
        print("✅ SUCCÈS : Inférence déterministe.")
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant Train/Test")
    parser.add_argument('--weights', type=str, required=True, help="Dossier contenant les époques du modèle (ex: modelWeights/DeepLab/)")
    parser.add_argument('--model_name', type=str, required=True, choices=['UNet', 'ResUNet', 'ResUNetDS', 'DeepLab', 'SegNet'], help="Nom du modèle à évaluer")
    args = parser.parse_args()

    CONF['Model'] = args.model_name
    CONF['l2'] = MODEL_L2.get(args.model_name, 1e-9)

    _, _, d_val, g_val = load_dataset(args.input_dir)
    if len(d_val) == 0:
        return

    epoch_dirs = [os.path.join(args.weights, d) for d in os.listdir(args.weights) if os.path.isdir(os.path.join(args.weights, d)) and "epoch" in d]
    
    def extract_epoch_num(path):
        import re
        match = re.search(r'epoch_?(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else -1
        
    epoch_dirs.sort(key=extract_epoch_num)

    wandb.init(
        project="chronoRoot_logs",
        config=CONF,
        name=f"CLDICE_{args.model_name}_re_eval"
    )

    tf.compat.v1.reset_default_graph()
    config_proto = tf.compat.v1.ConfigProto()
    config_proto.gpu_options.allow_growth = True
    
    log_path_val = os.path.join(CONF['logDirRoot'], args.model_name, 'post_train_eval')
    os.makedirs(log_path_val, exist_ok=True)

    with tf.compat.v1.Session(config=config_proto) as sess:
        print(f"Construction du graphe pour {args.model_name}...")
        net = RootNet(sess, CONF, args.model_name, isTrain=True) 
        val_writer = tf.compat.v1.summary.FileWriter(log_path_val)

        for epoch_dir in epoch_dirs:
            epoch = extract_epoch_num(epoch_dir)
            if epoch == -1:
                continue
                
            print(f"\n=============================================")
            print(f" Epoch: {epoch_dir}")
            print(f"=============================================")
            
            try:
                net.restore(epoch_dir)
            except Exception as _:
                print(f"⚠️  Unable to restore from {epoch_dir}, skipping...")
                continue
            if epoch % 20 == 0 or epoch == 1:
                is_deterministic = check_reproducibility(net, d_val, args.model_name)
                if not is_deterministic:
                    print("Ending evaluation due to stochastic divergence.")
                    return

            do_heavy = False 
            val_metrics = evaluate_validation(sess, net, d_val, g_val, CONF, val_writer, epoch, args.model_name, do_heavy, use_crf=True)

            wandb_metrics = {f"val/{k}": v for k, v in val_metrics.items()}
            wandb_metrics["epoch"] = epoch
            wandb.log(wandb_metrics)
            
        val_writer.flush()
        
    wandb.finish()


if __name__ == "__main__":
    main()