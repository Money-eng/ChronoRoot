import tensorflow as tf
import os
import numpy as np
import pathlib
import re
import cv2
import argparse
import pydensecrf.densecrf as dcrf
import nibabel as nib 
from rootNet.Model import RootNet
from rootNet.Provider import DataProvider

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# --- Helpers ---

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def loadPath(search_path, ext='*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key=natural_key)
    return all_files

def mkdir(dir_path):
    try: os.makedirs(dir_path)
    except: pass 

def save_image_with_scale(path, arr):
    arr = np.clip(arr, 0., 1.)
    arr = arr * 255.
    arr = arr.astype(np.uint8)
    cv2.imwrite(path, arr)

def SaveSegImage(conf, name, segmentation, path, suffix=".png", cutpad=False):    
    if cutpad and 'OriginalSize' in conf:
        h, w = conf['OriginalSize']
        segmentation = segmentation[:h, :w]

    if suffix == ".nii.gz":
        name = name[0][0].replace(suffix, ".nii.gz")
        nombre = os.path.join(path, name)
        img = nib.Nifti1Image(segmentation.transpose(), np.eye(4))
        nib.save(img, nombre)
    else:
        # On garde le nom original du fichier
        base_name = os.path.basename(name[0][0])
        nombre = os.path.join(path, base_name)
        save_image_with_scale(nombre, segmentation)
    return

# --- Core Functions ---

def Segment(net, conf, input_dir, output_dir, use_crf):
    """
    Segmente un sous-dossier (rpi...) avec un modèle déjà chargé (net).
    Version fournie par l'utilisateur : Pas d'accum, sauvegarde directe.
    """
    Provider = DataProvider(input_dir, data_suffix=".png")
    if len(Provider.data_files) == 0: return

    data, name = Provider(1)
    
    # Note : La session est gérée à l'extérieur maintenant
    conf["batchSize"] = 1
    conf["tileSize"] = list(data.shape[1:3])

    limit = conf['LIMIT']

    if limit != -1:
        n = limit
    else:
        n = len(Provider.data_files)

    for i in range(0, n):
        if i != 0:
            data, name = Provider(1)
            
        segment = net.segment(data)

        if use_crf:
            image = cv2.cvtColor((data[0,:,:,0]*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
            image = np.ascontiguousarray(image)
            label_1 = np.transpose(segment[0,:,:,:], (2,0,1))
            unary = -np.log(np.clip(label_1,1e-5,1.0))
            c, h, w = unary.shape
            unary = unary.transpose(0, 2, 1)
            unary = unary.reshape(2, -1)
            unary = np.ascontiguousarray(unary)
            denseCRF = dcrf.DenseCRF2D(w, h, 2)
            denseCRF.setUnaryEnergy(unary)  
            denseCRF.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image, compat=1)
            q = denseCRF.inference(1)
            crf_map = np.array(q).reshape(2, w, h).transpose(2, 1, 0)
            
            out = crf_map[:,:,1]
        else:
            out = segment[0,:,:,1]
        
        SaveSegImage(conf, name, out, output_dir, ".png", True)


def compute_ensemble_for_subfolder(conf, input_subdir_path, output_epoch_root, available_models, sub_folder_name, use_crf):
    """
    Combine les résultats des différents modèles ET applique l'accumulation temporelle.
    """
    # Dossier de sortie final pour l'ensemble
    final_output_dir = os.path.join(output_epoch_root, 'Ensemble', sub_folder_name)
    mkdir(final_output_dir)
    
    # Chemins vers les résultats individuels des modèles (directement dans le dossier du modèle)
    model_output_dirs = [os.path.join(output_epoch_root, m, sub_folder_name) for m in available_models]
    
    # Liste des images originales pour l'ordre temporel
    images_orig = loadPath(input_subdir_path, '*.png')
    n = len(images_orig)
    
    if n == 0: return

    # Accumulateur pour l'ensemble temporel (ChronoRoot logic)
    first_img_shape = cv2.imread(images_orig[0], 0).shape
    accum = np.zeros(first_img_shape, dtype=float)

    print(f"    Computing Ensemble + Time for {sub_folder_name}...")

    for i in range(n):
        filename = os.path.basename(images_orig[i])
        
        segs = []
        for m_dir in model_output_dirs:
            mask_path = os.path.join(m_dir, filename)
            
            if os.path.exists(mask_path):
                img = cv2.imread(mask_path, 0)
                if img is None:
                    # print(f"Warning: Failed to load {mask_path}")
                    segs.append(np.zeros(first_img_shape))
                else:
                    # Normalisation 0-255 -> 0.0-1.0 pour le calcul
                    segs.append(img.astype('float') / 255.0)
            else:
                segs.append(np.zeros(first_img_shape))
        
        # 1. Moyenne des modèles (Ensemble pur)
        if len(segs) > 0:
            ensemble_pred = np.mean(segs, axis=0)
        else:
            ensemble_pred = np.zeros(first_img_shape)

        # 2. Accumulation Temporelle (Ensemble + Time)
        # C'est ici qu'on applique la logique ChronoRoot puisque la fonction Segment ne le fait plus
        accum = conf['Alpha'] * accum + ensemble_pred
        
        # Clip pour sécurité avant threshold
        accum = np.clip(accum, 0.0, 1.0)
        
        # 3. Threshold Final
        _, outimg = cv2.threshold(accum, conf['Thresh'], 1.0, cv2.THRESH_BINARY)
        
        # Sauvegarde
        fake_name = [[filename]] 
        SaveSegImage(conf, fake_name, outimg, final_output_dir, ".png", True)


def run_ensemble_pipeline(conf, models_root, data_root, output_root, use_crf):
    
    # 1. Définir les modèles disponibles
    # Vous pouvez ajuster cette liste selon vos dossiers réels
    all_potential_models = ['ResUNetDS', 'UNet', 'SegNet', 'DeepLab']
    available_models = [m for m in all_potential_models if os.path.exists(os.path.join(models_root, m))]
    
    if not available_models:
        raise Exception("Aucun modèle trouvé dans models_root (vérifiez les noms de dossiers)")

    print(f"Models found: {available_models}")

    # 2. Trouver les epochs
    ref_model_dir = os.path.join(models_root, available_models[0])
    epoch_dirs = [d for d in os.listdir(ref_model_dir) if "epoch_" in d and os.path.isdir(os.path.join(ref_model_dir, d))]
    epoch_dirs.sort(key=lambda s: int(re.search(r'\d+', s).group()))
    
    print(f"Epochs to process: {epoch_dirs}")

    # 3. Trouver les sous-dossiers de données
    data_subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    
    # --- MAIN LOOP ---
    for epoch in epoch_dirs:
        print(f"\n=== Processing {epoch} ===")
        output_epoch_root = os.path.join(output_root, epoch)
        
        # A. GENERATION INDIVIDUELLE
        for model_name in available_models:
            print(f"  > Generating masks for {model_name}...")
            
            # Reset graph pour chaque modèle pour éviter conflits de variables
            tf.compat.v1.reset_default_graph() 
            sess = tf.compat.v1.Session()
            
            try:
                ckpt_path = os.path.join(models_root, model_name, epoch)
                conf['Model'] = model_name 
                
                if not os.path.exists(ckpt_path):
                    print(f"    Skipping {model_name} (checkpoint missing for {epoch})")
                    continue
                
                # Initialisation du réseau
                conf["batchSize"] = 1
                conf["tileSize"] = [256, 256] 
                
                # Création du réseau
                try:
                    net = RootNet(sess, conf, "RootNET", False)
                    net.restore(ckpt_path)
                except Exception as e:
                    print(f"    Error initializing {model_name}: {e}")
                    continue

                for sub_data in data_subdirs:
                    input_subdir_path = os.path.join(data_root, sub_data)
                    output_model_subdir = os.path.join(output_epoch_root, model_name, sub_data)
                    mkdir(output_model_subdir)
                    
                    # Si le dossier est déjà rempli, on peut skip (optionnel)
                    # if len(os.listdir(output_model_subdir)) > 0: continue

                    Segment(net, conf, input_subdir_path, output_model_subdir, use_crf)
            
            except Exception as e:
                print(f"Error processing model {model_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                sess.close()

        # B. AGGREGATION (ENSEMBLE)
        print(f"  > Computing Ensemble for {epoch}...")
        for sub_data in data_subdirs:
            input_subdir_path = os.path.join(data_root, sub_data)
            try:
                compute_ensemble_for_subfolder(conf, input_subdir_path, output_epoch_root, available_models, sub_data, use_crf)
            except Exception as e:
                print(f"Error ensemble {sub_data}: {e}")

    print("\nProcessing Complete.")

if __name__ == "__main__":
    conf1 = {}
    try: exec(open('config.conf').read(), conf1)
    except: conf1 = {'LIMIT': -1, 'OriginalSize': [100,100], 'Alpha': 1.0, 'Thresh': 0.5}
        
    conf2 = {}
    try: exec(open('cnns.conf').read(), conf2)
    except: pass
    
    conf = {**conf1, **conf2}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_root', required=True, help="Dossier contenant les dossiers des modèles (UNet, ResUNetDS...)")
    parser.add_argument('--data_root', required=True, help="Dossier racine des données (rpi...)")
    parser.add_argument('--output_dir', required=True, help="Dossier racine de sortie")
    parser.add_argument('--use_crf', action='store_true', default=False)

    args = parser.parse_args()
    
    mkdir(args.output_dir)

    run_ensemble_pipeline(conf, args.models_root, args.data_root, args.output_dir, args.use_crf)