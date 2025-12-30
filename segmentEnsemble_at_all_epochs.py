"""
ChronoRoot: High-throughput phenotyping by deep learning 
(Optimisé : Float64 + Skip intelligent si les dossiers sont complets)
"""

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
from tensorflow.python.util import deprecation
from multiprocessing import Pool
from functools import partial

from rootNet.Model import RootNet
from rootNet.Provider import DataProvider

deprecation._PRINT_DEPRECATION_WARNINGS = False

def natural_key(string_):
    """Tri naturel (ex: 2 avant 10)"""
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
    """
    Vérifie RAPIDEMENT si tous les fichiers de sortie attendus existent déjà.
    """
    if not os.path.exists(output_dir):
        return False

    # 1. Lister les entrées attendues
    inputs = loadPath(input_dir, '*.png')
    if not inputs:
        return False
    
    if limit != -1:
        inputs = inputs[:limit]

    # 2. Générer les noms de fichiers attendus en sortie (Set pour recherche O(1))
    expected_files = {os.path.basename(f).replace('.png', output_suffix) for f in inputs}
    
    # 3. Lister les fichiers présents dans la sortie
    present_files = set(os.listdir(output_dir))

    # 4. Vérifier si tout ce qu'on attend est présent (subset)
    # Renvoie True si expected_files est inclus dans present_files
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
    
    # --- CHECK DE SKIP INTELLIGENT ---
    # Si tous les fichiers sont déjà là, on ne charge même pas TensorFlow
    if is_task_completed(input_dir, output_dir, "_mask.png", conf.get('LIMIT', -1)):
        print(f"      [SKIP COMPLET] Segmentation déjà terminée pour {os.path.basename(output_dir)}")
        return
    # ---------------------------------

    Provider = DataProvider(input_dir, data_suffix=".png")
    data, name = Provider(1)
    
    sess = tf.compat.v1.Session()

    conf["batchSize"] = 1
    conf["tileSize"] = list(data.shape[1:3])

    net = RootNet(sess, conf, "RootNET", False)
    
    if checkpoint_path:
        conf['ckptDir'] = checkpoint_path
    else:
        conf['ckptDir'] = os.path.join(os.path.join('modelWeights', conf['Model']), 'ckpt')
    
    try:
        net.restore(conf['ckptDir'])
    except Exception as e:
        print(f"      [ERREUR] Impossible de charger les poids : {e}")
        sess.close()
        return

    limit = conf['LIMIT']
    n = limit if limit != -1 else len(Provider.data_files)
    
    for i in range(0, n):
        if i != 0:
            data, name = Provider(1)
            
        mask_name = name[0][0].replace('.png', '_mask.png')
        # Check unitaire au cas où le dossier serait partiellement rempli
        if os.path.exists(os.path.join(output_dir, mask_name)):
            # print(f"      [Saut] Masque déjà existant pour {mask_name}")
            continue
            
        segment = net.segment(data)
        outimg = segment[0,:,:,1]
        
        SaveSegImage(conf, name, outimg, output_dir, ".png", True)
        
    tf.compat.v1.reset_default_graph()
    sess.close()    

def process_single_frame(image_path, model_paths, valid_models, use_crf):
    filename = os.path.basename(image_path)
    mask_name = filename.replace('.png', '_mask.png')
    
    original_img = cv2.imread(image_path, 0)
    if original_img is None:
        return None 
    
    shape = original_img.shape
    segs = []

    for mp in model_paths:
        full_mask_path = os.path.join(mp, mask_name)
        if os.path.exists(full_mask_path):
            segs.append(cv2.imread(full_mask_path, 0).astype('float64') / 255.0)
        else:
            segs.append(np.zeros(shape, dtype=np.float64))

    if not segs:
        return np.zeros(shape, dtype=np.float64)
        
    ensemble = np.mean(segs, axis=0) 
    
    if use_crf:
        image_rgb = cv2.imread(image_path, 1) 
        image_rgb = np.ascontiguousarray(image_rgb)
        
        ensemble_2ch = np.dstack((1.0 - ensemble, ensemble))
        label_1 = np.transpose(ensemble_2ch, (2, 0, 1))
        
        unary = -np.log(np.clip(label_1, 1e-5, 1.0))
        unary = unary.astype(np.float64) # Float64 ici pour calcul interne
        
        C, H, W = unary.shape
        unary = unary.reshape(2, -1)
        unary = np.ascontiguousarray(unary)
        
        # Le binding C++ de pydensecrf attend généralement du float32 pour l'input
        # On cast juste avant l'envoi
        denseCRF = dcrf.DenseCRF2D(W, H, 2)
        denseCRF.setUnaryEnergy(unary.astype(np.float32)) 
        denseCRF.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image_rgb, compat=1)
        
        q = denseCRF.inference(1)
        crf_map = np.array(q).reshape(2, H, W)
        return crf_map[1, :, :].astype(np.float64) # Retour en float64
    else:
        return ensemble
    
def ensembleModels(conf, input_dir, output_root_for_epoch, crf, models, sub_folder=""):   
    ensemble_final_dir = os.path.join(output_root_for_epoch, "EnsembleResult", sub_folder)
    
    # --- CHECK DE SKIP INTELLIGENT ---
    if is_task_completed(input_dir, ensemble_final_dir, "_ensemble.png", conf.get('LIMIT', -1)):
        print(f"      [SKIP COMPLET] Ensemble déjà terminé pour {sub_folder}")
        return
    # ---------------------------------

    model_paths = []
    valid_models = []
    for m in models:
        p = os.path.join(output_root_for_epoch, m, sub_folder)
        if os.path.exists(p):
            model_paths.append(p)
            valid_models.append(m)
    
    if not valid_models:
        return

    images = loadPath(input_dir, '*.png') 
    if not images:
        return

    limit = conf['LIMIT']
    n = limit if limit != -1 else len(images)
    images = images[:n]

    mkdir(ensemble_final_dir)

    h, w = cv2.imread(images[0], 0).shape
    accum = np.zeros((h, w), dtype=np.float64)

    worker = partial(process_single_frame, 
                     model_paths=model_paths, 
                     valid_models=valid_models, 
                     use_crf=crf)

    num_processes = min(os.cpu_count(), 32) 
    print(f"      [Ensemble] Démarrage parallèle sur {num_processes} coeurs...")

    with Pool(processes=num_processes) as pool:
        for i, processed_img in enumerate(pool.imap(worker, images, chunksize=1)): 
            
            image_path = images[i] 
            
            if processed_img is None:
                print(f"      [Erreur] Image {os.path.basename(image_path)} ignorée.")
                continue

            accum = (conf['Alpha'] * accum + processed_img) / (1.0 + conf['Alpha'])
            
            _, outimg = cv2.threshold(accum, conf['Thresh'], 1.0, cv2.THRESH_BINARY)
            
            filename = os.path.basename(image_path)
            final_name = filename.replace('.png', '_ensemble.png')
            save_image_with_scale(os.path.join(ensemble_final_dir, final_name), outimg)
            
            if i % 50 == 0:
                print(f"      Traitement: {i}/{n}", end='\r')

    print(f"      [Ensemble] Terminé pour {sub_folder}")
    return

if __name__ == "__main__":
    conf1 = {}
    try:
        exec(open('config.conf').read(), conf1)
        conf2 = {}
        exec(open('cnns.conf').read(), conf2)
    except FileNotFoundError:
        print("Erreur: Fichiers de configuration introuvables.")
        exit(1)
    
    conf = {**conf1, **conf2}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_crf', action='store_true', default=True, help='Apply CRF post-processing')
    parser.add_argument('--output_dir', type=str, help='Output directory root', nargs="?")
    parser.add_argument('--input_dir', type=str, help='Input directory', nargs="?")

    args = parser.parse_args()
    
    if not args.input_dir:
        parser.print_help()
        raise Exception("Input directory required")
    
    base_output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, 'SegEnsemble_AllEpochs')
    mkdir(base_output_dir)

    use_crf = args.use_crf
    available_models = ['DeepLab', 'ResUNet', 'ResUNetDS', 'SegNet', 'UNet']

    reference_model_dir = os.path.join('modelWeights', 'DeepLab')
    if not os.path.exists(reference_model_dir):
        reference_model_dir = os.path.join('modelWeights', available_models[1])

    epoch_folders = [d for d in os.listdir(reference_model_dir) if d.startswith('epoch_') and os.path.isdir(os.path.join(reference_model_dir, d))]
    epoch_folders.sort(key=natural_key, reverse=True)
    print(f"Époques détectées : {len(epoch_folders)}")

    all_content = os.listdir(args.input_dir)
    sub_dirs = [d for d in all_content if os.path.isdir(os.path.join(args.input_dir, d)) and not d.startswith('.')]
    sub_dirs.sort(key=natural_key)
    
    # --- 3. Boucle Principale ---
    for epoch_name in epoch_folders:
        print(f"\n=== Traitement de : {epoch_name} ===")
        
        epoch_output_root = os.path.join(base_output_dir, epoch_name)
        mkdir(epoch_output_root)
        
        epoch_number = epoch_name.split('_')[1]
        epoch_number = int(epoch_number)
        if (epoch_number % 5) != 0:
            continue 

        for sub in sub_dirs:
            current_input_dir = os.path.join(args.input_dir, sub)
            print(f" -> Dossier Image : {sub if sub else 'Root'}")

            for model_name in available_models:
                conf['Model'] = model_name  
                
                model_sub_out_dir = os.path.join(epoch_output_root, model_name, sub)
                mkdir(model_sub_out_dir)

                weights_path = os.path.join('modelWeights', model_name, epoch_name)
                
                if os.path.exists(weights_path): 
                    # L'appel à Segment inclut maintenant le skip automatique
                    Segment(conf, current_input_dir, model_sub_out_dir, checkpoint_path=weights_path)
                else:
                    pass
            
            # L'appel à ensembleModels inclut maintenant le skip automatique
            ensembleModels(conf, current_input_dir, epoch_output_root, use_crf, available_models, sub_folder=sub)
            
        print(f"      [Nettoyage] Suppression des résultats intermédiaires pour {epoch_name}...")
        for model_name in available_models:
            folder_to_delete = os.path.join(epoch_output_root, model_name)
            if os.path.exists(folder_to_delete):
                try:
                    shutil.rmtree(folder_to_delete)
                except Exception as e:
                    print(f"      [Erreur] Impossible de supprimer {folder_to_delete} : {e}")