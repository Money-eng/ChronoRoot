import tensorflow as tf
import os
import numpy as np
import glob
import re

from rootNet.Model import RootNet
from rootNet.Provider import DataProvider

import nibabel as nib
import cv2
import argparse
import pydensecrf.densecrf as dcrf

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False


def mkdir(dir_path):
    try:
        os.makedirs(dir_path)
    except:
        pass


def save_image_as_it_is(path, arr):
    cv2.imwrite(path, arr)


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
        base_name = os.path.basename(name[0][0])
        output_name = base_name 
        nombre = os.path.join(path, output_name)
        save_image_with_scale(nombre, segmentation)
    return


def SegmentUNet(net, conf, input_dir, output_dir, use_crf):
    """
    Exécute la segmentation.
    Note: On ne passe plus checkpoint_dir ici car le restore est fait avant.
    """
    # 1. Setup Provider
    Provider = DataProvider(input_dir, data_suffix=".png")

    if len(Provider.data_files) == 0:
        print(f"Skipping empty folder: {input_dir}")
        return

    data, name = Provider(1)
    limit = conf['LIMIT']
    n = limit if limit != -1 else len(Provider.data_files)

    accum = np.zeros(data.shape[1:3], dtype=np.float32)

    for i in range(0, n):
        if i != 0:
            data, name = Provider(1)

        segment = net.segment(data)

        if use_crf:
            image = cv2.cvtColor(
                (data[0, :, :, 0]*255).astype('uint8'), cv2.COLOR_GRAY2RGB)
            image = np.ascontiguousarray(image)

            label_1 = np.transpose(segment[0, :, :, :], (2, 0, 1))
            unary = -np.log(np.clip(label_1, 1e-5, 1.0))
            c, h, w = unary.shape
            unary = unary.transpose(0, 2, 1)
            unary = unary.reshape(2, -1)
            unary = np.ascontiguousarray(unary)

            denseCRF = dcrf.DenseCRF2D(w, h, 2)
            denseCRF.setUnaryEnergy(unary)
            denseCRF.addPairwiseBilateral(sxy=5, srgb=3, rgbim=image, compat=1)

            q = denseCRF.inference(1)
            crf_map = np.array(q).reshape(2, w, h).transpose(2, 1, 0)

            accum = conf['Alpha'] * accum + crf_map[:, :, 1]
        else:
            accum = conf['Alpha'] * accum + segment[0, :, :, 1]
            
        # save accum as float image for debugging
        #accum_path = os.path.join(output_dir, os.path.basename(name[0][0]).replace(".png", "_accum.tiff"))
        #cv2.imwrite(accum_path, accum.astype(np.float32))
        
        _, outimg = cv2.threshold(accum, conf['Thresh'], 1.0, cv2.THRESH_BINARY)
        nameimg = name

        output_dir_img = os.path.join(output_dir, "accum_images")
        os.makedirs(output_dir_img, exist_ok=True)
        SaveSegImage(conf, name, outimg, output_dir_img, ".png", True)
        
        nameseg = name
        _, outseg = cv2.threshold(segment[0, :, :, 1], conf['Thresh'], 1.0, cv2.THRESH_BINARY)
        output_dir_seg = os.path.join(output_dir, "segmentation_images")
        os.makedirs(output_dir_seg, exist_ok=True)
        SaveSegImage(conf, name, outseg, output_dir_seg, ".png", True)

def run_multi_epoch_segmentation(conf, models_root, data_root, output_root, model_name, use_crf):
    """
    Parcourt les epochs et les sous-dossiers de données.
    """

    # 1. Trouver les epochs
    model_base_dir = os.path.join(models_root, model_name)
    if not os.path.exists(model_base_dir):
        raise Exception(f"Model directory not found: {model_base_dir}")

    epoch_dirs = [d for d in os.listdir(model_base_dir)
                  if os.path.isdir(os.path.join(model_base_dir, d)) and "epoch_" in d]

    epoch_dirs.sort(key=lambda s: int(re.search(r'\d+', s).group()))

    print(f"Found {len(epoch_dirs)} epochs to process: {epoch_dirs}")

    # 2. Trouver les dossiers de données
    data_subdirs = [d for d in os.listdir(data_root)
                    if os.path.isdir(os.path.join(data_root, d))]

    # 3. Boucle principale
    for epoch in epoch_dirs:
        print(f"\n================ STARTING {epoch} ================")

        # --- CORRECTION MAJEURE ICI ---
        # On réinitialise le graphe TensorFlow AVANT de créer une nouvelle session et un nouveau RootNet
        tf.compat.v1.reset_default_graph() 
        sess = tf.compat.v1.Session()
        # ------------------------------

        current_ckpt_dir = os.path.join(model_base_dir, epoch)
        current_epoch_output = os.path.join(output_root, model_name, epoch)
        
        conf["batchSize"] = 1
        conf["tileSize"] = [256, 256] # Assurez-vous que cela correspond à vos images
        
        # Initialisation du modèle (crée les variables dans le graphe vide)
        try:
            net = RootNet(sess, conf, "RootNET", False)
            net.restore(current_ckpt_dir)
            
            for sub_data in data_subdirs:
                input_subdir_path = os.path.join(data_root, sub_data)
                output_subdir_path = os.path.join(current_epoch_output, sub_data)
                mkdir(output_subdir_path)

                print(f"Processing {sub_data}...")
                SegmentUNet(net, conf, input_subdir_path, output_subdir_path, use_crf)
                
        except Exception as e:
            print(f"ERROR processing {epoch}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # On ferme la session à la fin de l'epoch pour libérer la mémoire GPU/CPU
            sess.close()


if __name__ == "__main__":
    conf1 = {}
    try:
        file = exec(open('config.conf').read(), conf1)
    except:
        pass #conf1 = {'LIMIT': -1, 'OriginalSize': [100, 100], 'Alpha': 1.0, 'Thresh': 0.5}

    conf2 = {}
    try:
        file = exec(open('cnns.conf').read(), conf2)
    except:
        pass

    conf = {**conf1, **conf2}

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default="ResUNetDS", help="Nom du modèle")
    parser.add_argument('--models_root', required=True, help="Dossier modèles")
    parser.add_argument('--data_root', required=True, help="Dossier données")
    parser.add_argument('--output_dir', required=True, help="Dossier output")
    parser.add_argument('--use_crf', action='store_true', default=False)

    args = parser.parse_args()
    conf['Model'] = args.model_name

    mkdir(args.output_dir)

    run_multi_epoch_segmentation(
        conf=conf,
        models_root=args.models_root,
        data_root=args.data_root,
        output_root=args.output_dir,
        model_name=args.model_name,
        use_crf=args.use_crf
    )
    
    print("\nJob Done.")