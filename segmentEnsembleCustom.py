import tensorflow as tf
import os
import numpy as np
import nibabel as nib
import cv2
import argparse
import pydensecrf.densecrf as dcrf
import shutil
import re
import pathlib
import multiprocessing as mp
from tensorflow.python.util import deprecation
from functools import partial

from rootNet.Model import RootNet
from rootNet.Provider import DataProvider
import gc

deprecation._PRINT_DEPRECATION_WARNINGS = False

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext='*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key=natural_key)
    return all_files

def mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except: pass 

def is_task_completed(input_dir, output_dir, output_suffix, limit=-1):
    if not os.path.exists(output_dir):
        return False
    inputs = loadPath(input_dir, '*.png')
    if not inputs: return False
    if limit != -1: inputs = inputs[:limit]
    expected_files = {os.path.basename(f).replace('.png', output_suffix) for f in inputs}
    present_files = set(os.listdir(output_dir))
    return expected_files.issubset(present_files)

def save_image_with_scale(path, arr):
    arr = np.clip(arr, 0., 1.)
    arr = arr * 255.
    arr = arr.astype(np.uint8)
    cv2.imwrite(path, arr)

def SaveSegImage(conf, name, segmentation, path, suffix=".png", cutpad=False):    
    if cutpad:
        h, w = conf['OriginalSize']
        segmentation = segmentation[:h, :w]
    if suffix == ".nii.gz":
        name = name[0][0].replace(suffix, ".nii.gz")
        nombre = os.path.join(path, name)
        img = nib.Nifti1Image(segmentation.transpose(), np.eye(4))
        nib.save(img, nombre)
    else:   
        name = name[0][0].replace(suffix, "_mask.png")
        nombre = os.path.join(path, name)
        save_image_with_scale(nombre, segmentation)
    return

def Segment(conf, input_dir, output_dir, checkpoint_path=None):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    
    if is_task_completed(input_dir, output_dir, "_mask.png", conf.get('LIMIT', -1)):
        return # Skip silent
    
    tf_config = tf.compat.v1.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    Provider = DataProvider(input_dir, data_suffix=".png")
    data, name = Provider(1)
    
    sess = tf.compat.v1.Session(config=tf_config)

    conf["batchSize"] = 1
    conf["tileSize"] = list(data.shape[1:3])

    net = RootNet(sess, conf, "RootNET", False)
    
    ckpt = checkpoint_path if checkpoint_path else os.path.join('modelWeights', conf['Model'], 'ckpt')
    
    try:
        net.restore(ckpt)
    except Exception as e:
        print(f"      [ERREUR LOAD] {conf['Model']} : {e}")
        sess.close()
        tf.compat.v1.reset_default_graph()
        return

    limit = conf['LIMIT']
    n = limit if limit != -1 else len(Provider.data_files)
    
    for i in range(0, n):
        if i != 0: data, name = Provider(1)
        
        mask_name = name[0][0].replace('.png', '_mask.png')
        if os.path.exists(os.path.join(output_dir, mask_name)):
            continue
        
        try:
            segment = net.segment(data)
        except Exception as e:
            print(f"      [ERREUR SEGMENT] {conf['Model']} : {e}")
            continue
        outimg = segment[0,:,:,1]
        SaveSegImage(conf, name, outimg, output_dir, ".png", True)
        
    sess.close()
    tf.compat.v1.reset_default_graph()
    del net
    del sess
    del Provider
    del data
    gc.collect()

def process_single_frame(image_path, model_paths, valid_models, use_crf):
    filename = os.path.basename(image_path)
    mask_name = filename.replace('.png', '_mask.png')
    original_img = cv2.imread(image_path, 0)
    if original_img is None: return None 
    shape = original_img.shape
    segs = []

    for mp in model_paths:
        full_mask_path = os.path.join(mp, mask_name)
        if os.path.exists(full_mask_path):
            segs.append(cv2.imread(full_mask_path, 0).astype('float32') / 255.0)
        else:
            segs.append(np.zeros(shape, dtype=np.float32))

    if not segs: return np.zeros(shape, dtype=np.float32)
    
    ensemble = np.mean(segs, axis=0).astype(np.float32)
    
    if use_crf:
        image_rgb = cv2.imread(image_path, 1) 
        image_rgb = np.ascontiguousarray(image_rgb)
        
        ensemble_2ch = np.dstack((1.0 - ensemble, ensemble))
        label_1 = np.transpose(ensemble_2ch, (2, 0, 1))
        
        unary = -np.log(np.clip(label_1, 1e-5, 1.0)).astype(np.float32)
        C, H, W = unary.shape
        unary = np.ascontiguousarray(unary.reshape(2, -1))
        
        denseCRF = dcrf.DenseCRF2D(W, H, 2)
        denseCRF.setUnaryEnergy(unary) 
        denseCRF.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image_rgb, compat=1)
        q = denseCRF.inference(1)
        
        return np.array(q).reshape(2, H, W)[1, :, :].astype(np.float32)
    else:
        return ensemble

def ensembleModels(conf, input_dir, output_root_for_epoch, crf, models, sub_folder="", max_cpu=4):   
    ensemble_final_dir = os.path.join(output_root_for_epoch, "EnsembleResult", sub_folder)
    
    if is_task_completed(input_dir, ensemble_final_dir, "_ensemble.png", conf.get('LIMIT', -1)):
        return

    model_paths = []
    valid_models = []
    for m in models:
        p = os.path.join(output_root_for_epoch, m, sub_folder)
        if os.path.exists(p):
            model_paths.append(p)
            valid_models.append(m)
    
    if not valid_models: return

    images = loadPath(input_dir, '*.png') 
    if not images: return
    limit = conf['LIMIT']
    if limit != -1: images = images[:limit]

    mkdir(ensemble_final_dir)
    h, w = cv2.imread(images[0], 0).shape
    accum = np.zeros((h, w), dtype=np.float32)

    worker = partial(process_single_frame, model_paths=model_paths, valid_models=valid_models, use_crf=crf)

    chunksize = 1 # max(1, len(images) // (max_cpu * 4))
    with mp.Pool(processes=max_cpu) as pool:
        for i, processed_img in enumerate(pool.imap(worker, images, chunksize=chunksize)): 
            if processed_img is None: continue
            accum = (conf['Alpha'] * accum + processed_img) / (1.0 + conf['Alpha'])
            _, outimg = cv2.threshold(accum, conf['Thresh'], 1.0, cv2.THRESH_BINARY)
            final_name = os.path.basename(images[i]).replace('.png', '_ensemble.png')
            save_image_with_scale(os.path.join(ensemble_final_dir, final_name), outimg)

    return

def process_job_folder(gpu_id, job_data, conf, available_models, base_output_dir, use_crf, max_cpu_allowed=96):
    epoch_name, sub_dir, input_dir_root = job_data
    
    current_input_dir = os.path.join(input_dir_root, sub_dir)
    epoch_output_root = os.path.join(base_output_dir, epoch_name)
    
    # print(f"[GPU {gpu_id}] Start: {epoch_name} / {sub_dir}")

    for model_name in available_models:
        conf['Model'] = model_name
        model_sub_out_dir = os.path.join(epoch_output_root, model_name, sub_dir)
        mkdir(model_sub_out_dir)
        
        weights_path = os.path.join('modelWeights', model_name, epoch_name)
        if os.path.exists(weights_path):
            Segment(conf, current_input_dir, model_sub_out_dir, checkpoint_path=weights_path)

    ensembleModels(conf, current_input_dir, epoch_output_root, use_crf, available_models, 
                   sub_folder=sub_dir, max_cpu=max_cpu_allowed)
    
    # empty memory 
    gc.collect()
    
    # delete model folders to save space
    for model_name in available_models:
        folder_to_delete = os.path.join(epoch_output_root, model_name, sub_dir)
        if os.path.exists(folder_to_delete):
            try: shutil.rmtree(folder_to_delete)
            except: pass
            
    # archive entire ensemble folder
    ensemble_folder = os.path.join(epoch_output_root, "EnsembleResult")
    tar_path = os.path.join(base_output_dir, 'ArchivedEpochs', f"{epoch_name}.tar.gz")
    if os.path.exists(ensemble_folder):
        shutil.make_archive(base_name=tar_path.replace('.tar.gz', ''), format='gztar', root_dir=ensemble_folder)
        try: shutil.rmtree(ensemble_folder)
        except: pass
    
    print(f"[GPU {gpu_id}] Terminé: {epoch_name}/{sub_dir}")


def gpu_worker(gpu_id, job_queue, conf, available_models, base_output_dir, use_crf, max_cpu_allowed=96):
    """
    Main loop for a persistent GPU worker process.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        try:
            job = job_queue.get()
            if job is None:
                break # Termination signal
            
            process_job_folder(gpu_id, job, conf, available_models, base_output_dir, use_crf, max_cpu_allowed)
            
        except Exception as e:
            print(f"[GPU {gpu_id}] CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    conf1 = {}
    try:
        exec(open('config.conf').read(), conf1)
        conf2 = {}
        exec(open('cnns.conf').read(), conf2)
    except FileNotFoundError:
        print("Erreur: Fichiers de configuration introuvables.")
        exit(1)
    
    conf = {**conf1, **conf2}
    
    conf.pop('__builtins__', None)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_crf', action='store_true', default=True)
    parser.add_argument('--output_dir', type=str, nargs="?")
    parser.add_argument('--input_dir', type=str, nargs="?")
    parser.add_argument('--gpus', type=str, default="0,1,2,3", help="List of GPUs to use, e.g. 0,1,2,3")

    args = parser.parse_args()
    
    if not args.input_dir:
        raise Exception("Input directory required")
    
    base_output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, 'SegEnsemble_AllEpochs')
    mkdir(base_output_dir)

    # Detect Available Epochs
    available_models = ['ResUNet', 'DeepLab', 'ResUNetDS', 'SegNet', 'UNet']
    reference_model_dir = os.path.join('modelWeights', 'DeepLab')
    if not os.path.exists(reference_model_dir):
        reference_model_dir = os.path.join('modelWeights', available_models[1])

    epoch_folders = [d for d in os.listdir(reference_model_dir) if d.startswith('epoch_')]
    epoch_folders.sort(key=natural_key, reverse=True)

    all_content = os.listdir(args.input_dir)
    sub_dirs = [d for d in all_content if os.path.isdir(os.path.join(args.input_dir, d)) and not d.startswith('.')]
    sub_dirs.sort(key=natural_key)

    job_queue = mp.Queue()
    
    os.makedirs(os.path.join(base_output_dir, 'ArchivedEpochs'), exist_ok=True)
    
    total_jobs = 0
    for epoch_name in epoch_folders:
        epoch_number = int(epoch_name.split('_')[1])
        if (epoch_number % 5) != 0: continue
        
        missing_model = None
        for model in available_models:
            weight_path = os.path.join('modelWeights', model, epoch_name)
            
            if not os.path.exists(weight_path) or len(os.listdir(weight_path)) == 0:
                missing_model = model
                break
        
        if missing_model:
            print(f"  [SKIP] {epoch_name} ignorée : Poids manquants pour '{missing_model}'")
            continue
        
        # if a tar.gz of this epoch exists, skip processing
        tar_path = os.path.join(base_output_dir, 'ArchivedEpochs', f"{epoch_name}.tar.gz")
        if os.path.exists(tar_path):
            print(f"  [SKIP] {epoch_name} ignorée : Archive trouvée.")
            continue
        
        mkdir(os.path.join(base_output_dir, epoch_name))
        
        for sub in sub_dirs:
            # Job data: (Epoch, SubFolder, InputRoot)
            job_queue.put( (epoch_name, sub, args.input_dir) )
            total_jobs += 1

    print(f"=== Workload: {total_jobs} tasks distributed on GPUs: {args.gpus} ===")
    
    # --- Start Workers ---
    gpu_list = [int(x) for x in args.gpus.split(',')]
    nb_gpus = len(gpu_list)
    total_cores = mp.cpu_count()
    
    processes = []
    
    cpus_per_worker = max(1, (total_cores - 2) // nb_gpus)
    
    for gpu_id in gpu_list:
        p = mp.Process(target=gpu_worker, 
                       args=(gpu_id, job_queue, conf, available_models, base_output_dir, args.use_crf, cpus_per_worker))
        p.start()
        processes.append(p)
        job_queue.put(None) 
    
    for p in processes:
        p.join()
        
        
    print("\nAll tasks completed.")