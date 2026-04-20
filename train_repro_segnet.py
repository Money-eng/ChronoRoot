import concurrent
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
import pydensecrf.densecrf as dcrf
from concurrent.futures import ProcessPoolExecutor, as_completed

CONF = {
    'tileSize': [256, 256],
    'batchSize': 8,
    'numEpochs': 200,
    'iterPerEpoch': 100,
    'learning_rate': 0.0001,
    'dropout': 0.30,
    'loss': 'cldice',  ######################" Change loss here ######################"""
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

MODEL_L2 = {
    'UNet': 1e-8, 'ResUNet': 1e-8, 'ResUNetDS': 1e-8, 'DeepLab': 1e-9, 'SegNet': 1e-10
}

HEAVY_METRICS_FREQ = 5

last_f1_score = 0.0


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

    train_dir = os.path.join(input_dir, 'Train')
    test_dir = os.path.join(input_dir, 'Test')

    if not os.path.exists(test_dir):
        if os.path.exists(os.path.join(input_dir, 'Validation')):
            test_dir = os.path.join(input_dir, 'Validation')
        elif os.path.exists(os.path.join(input_dir, 'val')):
            test_dir = os.path.join(input_dir, 'val')

    data_train, gt_train = load_folder(
        train_dir, img_suffix, mask_suffix, "Loading Train")
    data_val, gt_val = load_folder(
        test_dir, img_suffix, mask_suffix, "Loading Test")

    return data_train, gt_train, data_val, gt_val


def make_summary(name, value):
    return tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag=name, simple_value=float(value))])


def evaluate_validation(sess, net, data_val, gt_val, conf, writer, epoch, model_name, do_heavy, use_crf=True):
    global last_f1_score

    metrics_sum = {
        'loss': [], 'f1_score': [], 'f_2_score': [], 'f_3_score': [], 'f_4_score': [], 'precision': [], 'recall': [], 'specificity': [], 'mean_iou': [], 'iou': [], 'pixel_accuracy': [], 'dice': [], 'dice_gpu': [], 'auc_gpu': [], 'precision_gpu': [], 'recall_gpu': [], "hausdorff_95": [], "hausdorff_max": [], "surface_dice_1mm": [], "ASCD": [], "betti_0_abs_err": [], "betti_1_abs_err": [], "betti0_jaccard_ratio": [], "betti1_jaccard_ratio": [], "betti0_relative_error": [], "betti1_relative_error": [], "betti0_variation_index": [], "betti1_variation_index": [], "b0_pred": [], "b1_pred": [], 'normalized_mutual_information': []
    }

    if do_heavy:
        heavy_keys = ['apls', 'apls_recall', 'apls_precision']
        for k in heavy_keys:
            metrics_sum[k] = []

    # Constante pour SegNet
    MULTIPLE_OF = 32

    val_imgs_arr = np.array(data_val, dtype=np.float32)
    val_gts_arr = np.array(gt_val, dtype=np.float32)

    val_imgs_arr = val_imgs_arr[..., np.newaxis]

    val_imgs_arr = val_imgs_arr / 255.0

    MULTIPLE_OF = 32
    h, w = val_imgs_arr.shape[1], val_imgs_arr.shape[2]
    pad_h = (MULTIPLE_OF - (h % MULTIPLE_OF)) % MULTIPLE_OF
    pad_w = (MULTIPLE_OF - (w % MULTIPLE_OF)) % MULTIPLE_OF

    if pad_h > 0 or pad_w > 0:
        val_imgs_arr = np.pad(val_imgs_arr, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant')
        val_gts_arr = np.pad(val_gts_arr, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='constant')

    futures_list = []
    last_processed_pred = None
    last_processed_gt = None

    print(f"Validation Epoch {epoch} (Heavy={do_heavy})...")

    with concurrent.futures.ProcessPoolExecutor() as executor:

        for i, (img, gt) in tqdm.tqdm(enumerate(zip(val_imgs_arr, val_gts_arr)), total=len(val_imgs_arr),
                                      desc="GPU Inference"):
            img_input = img[np.newaxis, :, :, :]

            loss, diceg, aucg, precg, recg = net.deploy(img_input, gt[np.newaxis, :, :, :], phase=0)
            metrics_sum['loss'].append(loss)
            metrics_sum['dice_gpu'].append(diceg)
            metrics_sum['auc_gpu'].append(aucg)
            metrics_sum['precision_gpu'].append(precg)
            metrics_sum['recall_gpu'].append(recg)

            pred_prob_padded = net.segment(img_input)

            pred_prob = pred_prob_padded[:, :h, :w, :]
            img_crop = img[:h, :w, :]
            gt_crop = gt[:h, :w, :]
            future = executor.submit(
                process_single_validation_item,
                (pred_prob.copy(), img_crop.copy(), gt_crop.copy(), conf, use_crf, False)
            )
            futures_list.append(future)

        for future in tqdm.tqdm(as_completed(futures_list), total=len(futures_list), desc="CPU Metrics"):
            try:
                res, p_mask, g_mask = future.result()

                last_processed_pred = p_mask
                last_processed_gt = g_mask

                metrics_sum['loss'].append(0.0)
                for k, v in res.items():
                    if k in metrics_sum:
                        metrics_sum[k].append(v)

            except Exception as e:
                print(f"Erreur dans un worker : {e}")
                import traceback
                traceback.print_exc()

    avg_metrics = {k: np.mean(v) for k, v in metrics_sum.items() if v}

    print(f"\n--- Epoch {epoch} Validation Results ---")
    for k, v in avg_metrics.items():
        writer.add_summary(make_summary(f'val_{k}', v), epoch)
        print(f"  {k}: {v:.4f}")

    if last_processed_pred is not None and last_processed_gt is not None:
        save_dir = os.path.join(conf['logDirRoot'], f"model_{model_name}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(os.path.join(
            save_dir, f'val_pred_epoch_{epoch}.png'), last_processed_pred * 255)

    current_f1 = avg_metrics.get('f1', 0.0)
    last_f1_score = current_f1

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


def train_one_model(model_name, d_train, g_train, d_val, g_val):
    print(f"=== Entraînement : {model_name} ===")
    tf.compat.v1.reset_default_graph()

    current_conf = CONF.copy()
    current_conf['Model'] = model_name
    current_conf['l2'] = MODEL_L2.get(model_name, 1e-9)
    

    current_lr = current_conf['learning_rate']
    lr_patience = 10
    lr_factor = 0.5
    lr_min = 1e-6
    lr_wait = 0
    best_val_dice = float('-inf')

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

    load_from_last = False
    with tf.compat.v1.Session(config=config_proto) as sess:
        net = RootNet(sess, current_conf, model_name, isTrain=True)

        train_writer = tf.compat.v1.summary.FileWriter(log_path_train, sess.graph)
        val_writer = tf.compat.v1.summary.FileWriter(log_path_val)

        global_step = 0
        epoch_pbar = tqdm.tqdm(range(current_conf['numEpochs']), desc="Epochs", unit="ep")

        for epoch in epoch_pbar:
            epoch_save_dir = os.path.join(ckpt_base_path, f"epoch_{epoch + 1}")
            checkpoint_exists = False
            if os.path.exists(epoch_save_dir):
                if os.path.exists(os.path.join(epoch_save_dir, "checkpoint")):
                    checkpoint_exists = True

            if checkpoint_exists:
                tqdm.tqdm.write(f" -> Checkpoint for epoch {epoch + 1} already exists. Loading model...")
                load_from_last = True
                continue
                # net.restore(epoch_save_dir)
            if load_from_last and not checkpoint_exists:
                last_epoch = epoch - 1
                last_epoch_dir = os.path.join(ckpt_base_path, f"epoch_{last_epoch}")
                # assuming last epoch dir exists
                print(f" -> Chargement du checkpoint de l'époque {last_epoch}...")
                net.restore(last_epoch_dir)

            epoch_loss = 0.0
            batch_pbar = tqdm.tqdm(range(current_conf['iterPerEpoch']), desc=f"Epoch {epoch + 1}", leave=False)

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
                    loss = net.fit(batch_x, batch_y, learning_rate=current_lr, phase=True)

                    epoch_loss += loss
                    train_writer.add_summary(make_summary('batch_loss', loss), global_step)
                    global_step += 1

                    batch_pbar.set_postfix({'loss': f"{loss:.4f}", 'lr': f"{current_lr:.1e}"})

            avg_train_loss = epoch_loss / current_conf['iterPerEpoch']
            train_writer.add_summary(make_summary('epoch_loss', avg_train_loss), epoch)
            train_writer.add_summary(make_summary('learning_rate', current_lr), epoch)

            epoch_dir = os.path.join(ckpt_base_path, f"epoch_{epoch + 1}")
            os.makedirs(epoch_dir, exist_ok=True)

            if len(d_val) > 0:
                global last_f1_score
                do_heavy = False  # last_f1_score > 0.5 and ((epoch + 1) % HEAVY_METRICS_FREQ == 0) or ((epoch + 1) == current_conf['numEpochs'])

                metrics = evaluate_validation(sess, net, d_val, g_val, current_conf, val_writer, epoch, model_name,
                                              do_heavy)
                print(f"Validation metrics at the end of epoch {epoch + 1} : {metrics}")

                if not checkpoint_exists:
                    val_dice = metrics['dice']

                    if val_dice > (best_val_dice + 1e-4):
                        best_val_dice = val_dice
                        lr_wait = 0
                    else:
                        lr_wait += 1
                        print(f" -> Validation loss does not improve (Patience: {lr_wait}/{lr_patience})")

                        if lr_wait >= lr_patience:
                            old_lr = current_lr
                            current_lr = max(current_lr * lr_factor, lr_min)
                            lr_wait = 0
                            if current_lr < old_lr:
                                print(
                                    f"⚠️ Learning rate reduced from {old_lr:.1e} to {current_lr:.1e} due to plateau in validation performance.")

            epoch_pbar.set_postfix({'Train Loss': f"{avg_train_loss:.4f}", 'Val Loss': f"{metrics.get('loss', 0):.4f}"})

            print(f" -> Saving model checkpoint for epoch {epoch + 1}...")
            net.save(epoch_dir)
            train_writer.flush()
            val_writer.flush()

    batch_gen.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str,
                        default="./Data")
    parser.add_argument('--models', type=str, nargs='+',
                        default=['SegNet'])
    args = parser.parse_args()

    print(f"Starting training for models : {args.models}")

    d_train, g_train, d_val, g_val = load_dataset(args.input_dir)

    if len(d_train) == 0:
        print("Error: No data available.")
        return

    for model in args.models:
        print(f"\n\n=== Training the model : {model} ===")
        try:
            train_one_model(model, d_train, g_train,
                            d_val, g_val)
        except Exception as e:
            print(f"Error with {model}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
