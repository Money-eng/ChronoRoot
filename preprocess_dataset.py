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

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(".png") and not file.endswith("_mask.png") and "mask" not in file.lower():

                img_source_path = os.path.join(root, file)
                mask_nii_name = file.replace(".png", ".nii.gz")
                mask_nii_path = os.path.join(root, mask_nii_name)

                if not os.path.exists(mask_nii_path):
                    print(f"[SKIP] Masque introuvable pour : {os.path.relpath(img_source_path, input_root)}")
                    continue

                rel_path = os.path.relpath(root, input_root)
                prefix = rel_path.replace(os.sep, "_")
                if prefix == ".": prefix = ""

                candidate_name = f"{prefix}_{file}" if prefix else file

                unique_img_name = get_unique_filename(output_root, candidate_name)

                unique_mask_name = unique_img_name.replace(".png", "_mask.png")

                output_img_path = os.path.join(output_root, unique_img_name)
                output_mask_path = os.path.join(output_root, unique_mask_name)

                try:
                    shutil.copy2(img_source_path, output_img_path)

                    nii = nib.load(mask_nii_path)
                    mask_data = nii.get_fdata()

                    if len(mask_data.shape) == 3:
                        mask_2d = np.transpose(mask_data[:, :, 0])
                    else:
                        mask_2d = np.transpose(mask_data)

                    mask_uint8 = np.zeros_like(mask_2d, dtype=np.uint8)
                    mask_uint8[mask_2d > 0] = 255

                    cv2.imwrite(output_mask_path, mask_uint8)

                    processed_count += 1
                    print(f"[OK] {unique_img_name}", end='\r')

                except Exception as e:
                    error_msg = f"Erreur sur {file}: {str(e)}"
                    errors.append(error_msg)
                    if os.path.exists(output_img_path): os.remove(output_img_path)
                    if os.path.exists(output_mask_path): os.remove(output_mask_path)

    if errors:
        for err in errors:
            print(f"  - {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Dossier racine contenant les sous-dossiers')
    parser.add_argument('--output_dir', type=str, required=True, help='Dossier de destination plat')
    args = parser.parse_args()

    preprocess_and_flatten(args.input_dir, args.output_dir)
