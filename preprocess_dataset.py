import os
import shutil
import numpy as np
import nibabel as nib
import cv2
import argparse

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_unique_filename(directory, filename):
    """
    Génère un nom de fichier unique dans le dossier cible.
    Si 'fichier.png' existe, retourne 'fichier_1.png', etc.
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
        
    return new_filename

def preprocess_and_flatten(input_root, output_root):
    print(f"Source : {input_root}")
    print(f"Destination : {output_root}")
    
    ensure_dir(output_root)
    
    processed_count = 0
    errors = []
    
    # Parcourir récursivement tous les dossiers
    for root, dirs, files in os.walk(input_root):
        for file in files:
            # On cherche les images .png originales (qui ne sont pas des masques)
            if file.endswith(".png") and not file.endswith("_mask.png") and "mask" not in file.lower():
                
                img_source_path = os.path.join(root, file)
                
                # --- 1. Identification du Masque NIfTI ---
                # On suppose que le masque a le même nom mais finit par .nii.gz
                mask_nii_name = file.replace(".png", ".nii.gz")
                mask_nii_path = os.path.join(root, mask_nii_name)
                
                if not os.path.exists(mask_nii_path):
                    # Tenter une recherche plus souple si besoin (ex: rpi13_1.nii.gz vs rpi13_1.png)
                    print(f"[SKIP] Masque introuvable pour : {os.path.relpath(img_source_path, input_root)}")
                    continue

                # --- 2. Création d'un nom de base unique ---
                # On ajoute le dossier parent au nom pour aider (ex: rpi13_cam1_)
                rel_path = os.path.relpath(root, input_root)
                prefix = rel_path.replace(os.sep, "_")
                if prefix == ".": prefix = ""
                
                candidate_name = f"{prefix}_{file}" if prefix else file
                
                # VÉRIFICATION DES DOUBLONS ICI
                # On s'assure que le nom de l'image est unique dans le dossier de sortie
                unique_img_name = get_unique_filename(output_root, candidate_name)
                
                # Déduire le nom du masque basé sur le nom unique de l'image
                # Ex: si unique_img_name est "rpi13_cam1_img_2.png", le masque sera "rpi13_cam1_img_2_mask.png"
                unique_mask_name = unique_img_name.replace(".png", "_mask.png")
                
                output_img_path = os.path.join(output_root, unique_img_name)
                output_mask_path = os.path.join(output_root, unique_mask_name)
                
                try:
                    # --- 3. Traitement et Copie ---
                    
                    # A. Copie de l'image
                    shutil.copy2(img_source_path, output_img_path)
                    
                    # B. Conversion du masque NIfTI
                    nii = nib.load(mask_nii_path)
                    mask_data = nii.get_fdata()
                    
                    # Transposition (Vital pour aligner avec l'image PNG)
                    # Si le masque est vide ou dimension incorrecte, cela lèvera une erreur gérée plus bas
                    if len(mask_data.shape) == 3:
                        mask_2d = np.transpose(mask_data[:,:,0])
                    else:
                        mask_2d = np.transpose(mask_data)

                    # Binarisation et conversion uint8
                    # Tout ce qui n'est pas 0 devient 255 (Racine)
                    mask_uint8 = np.zeros_like(mask_2d, dtype=np.uint8)
                    mask_uint8[mask_2d > 0] = 255
                    
                    # Sauvegarde du masque
                    cv2.imwrite(output_mask_path, mask_uint8)
                    
                    processed_count += 1
                    print(f"[OK] {unique_img_name}", end='\r')
                    
                except Exception as e:
                    error_msg = f"Erreur sur {file}: {str(e)}"
                    errors.append(error_msg)
                    # Nettoyage en cas d'échec partiel (pour ne pas laisser une image sans masque)
                    if os.path.exists(output_img_path): os.remove(output_img_path)
                    if os.path.exists(output_mask_path): os.remove(output_mask_path)

    print(f"\n\n--- Résumé ---")
    print(f"Succès : {processed_count} paires générées dans '{output_root}'.")
    if errors:
        print(f"Échecs : {len(errors)}")
        for err in errors:
            print(f"  - {err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Dossier racine contenant les sous-dossiers')
    parser.add_argument('--output_dir', type=str, required=True, help='Dossier de destination plat')
    args = parser.parse_args()
    
    preprocess_and_flatten(args.input_dir, args.output_dir)