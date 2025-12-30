import cv2
import os
import numpy as np
from pathlib import Path

def compare_pixels_only(dir_ref, dir_new):
    dir_ref = Path(dir_ref)
    dir_new = Path(dir_new)
    
    print(f"comparaison visuelle (pixels uniquement) entre :\n 1. {dir_ref}\n 2. {dir_new}\n")
    
    # Récupérer la liste des images (png, jpg, tif...)
    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    files_ref = {p.relative_to(dir_ref) for p in dir_ref.rglob('*') if p.suffix.lower() in extensions}
    files_new = {p.relative_to(dir_new) for p in dir_new.rglob('*') if p.suffix.lower() in extensions}
    
    # Vérifier l'existence des fichiers
    missing_in_new = files_ref - files_new
    extra_in_new = files_new - files_ref
    
    if missing_in_new:
        print(f"❌ {len(missing_in_new)} fichiers manquants dans le nouveau dossier.")
    if extra_in_new:
        print(f"⚠️ {len(extra_in_new)} fichiers en trop dans le nouveau dossier (ignorés).")
        
    communs = files_ref & files_new
    print(f"🔍 Analyse de {len(communs)} images communes...")
    
    diff_count = 0
    error_count = 0
    
    for i, f in enumerate(communs):
        path1 = str(dir_ref / f)
        path2 = str(dir_new / f)
        
        # IMREAD_UNCHANGED est vital : il charge l'image telle quelle 
        # (avec canal Alpha transparence s'il existe, et en 16 bits si c'est du 16 bits)
        img1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)
        
        if img1 is None or img2 is None:
            print(f"⛔ Erreur de lecture : {f}")
            error_count += 1
            continue
            
        # 1. Comparaison des dimensions
        if img1.shape != img2.shape:
            print(f"❌ Différence de taille pour {f} : {img1.shape} vs {img2.shape}")
            diff_count += 1
            continue
            
        # 2. Comparaison des valeurs de pixels (STRICTE)
        # np.array_equal renvoie True si tous les éléments sont strictement identiques
        if not np.array_equal(img1, img2):
            # Calcul de la différence moyenne pour info
            diff = cv2.absdiff(img1, img2)
            non_zero = np.count_nonzero(diff)
            print(f"❌ Différence de pixels pour {f} ({non_zero} pixels différents)")
            diff_count += 1
        
        if i % 100 == 0:
            print(f"   Progression : {i}/{len(communs)}...", end='\r')

    print("\n" + "="*30)
    if diff_count == 0 and error_count == 0:
        print("✅ SUCCÈS TOTAL : Le contenu visuel (pixels) est 100% identique.")
    else:
        print(f"❌ ÉCHEC : {diff_count} images sont visuellement différentes.")

# --- LANCEZ LA COMPARAISON ICI ---
dossier_A = "/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/temp/new/1"
dossier_B = "/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/temp/old/1"

compare_pixels_only(dossier_A, dossier_B)