import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# ================= CONFIGURATION =================

# 1. Chemin racine des RSML (là où il y a ContLight/LongDay)
# D'après votre premier message
RSML_ROOT_DIR = Path("/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/loai/Data/ChronoRoot/RAW/Graphs/").expanduser() 

# 2. Chemin racine des IMAGES (là où il y a rpi6_..., rpi14_...)
# D'après votre second message
IMAGES_ROOT_DIR = Path("/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/loai/Data/ChronoRoot/RAW/Graph_inference").expanduser()

# 3. Dossier de sortie pour les MASQUES
# On va créer un dossier Masks à côté du dossier Graphs par exemple
OUTPUT_MASK_DIR = Path("/mnt/41d6c007-0c9e-41e2-b2eb-8d9c032e9e53/loai/Data/ChronoRoot/Mask_inference").expanduser()

# Diamètre de la racine en pixels (vous pouvez changer ça)
ROOT_THICKNESS = 5 

def parse_rsml_geometry(rsml_path):
    """ Extrait le nom de l'image cible et les polylignes """
    try:
        tree = ET.parse(rsml_path)
        root = tree.getroot()
        
        image_name_node = root.find(".//metadata/image/name")
        if image_name_node is None:
            return None, []
        image_name = image_name_node.text.strip()
        
        polylines = []
        # On récupère toutes les plantes, toutes les racines
        for plant in root.findall(".//scene/plant"):
            for root_entity in plant.findall(".//root"):
                polyline_node = root_entity.find(".//geometry/polyline")
                if polyline_node is not None:
                    points = []
                    for pt in polyline_node.findall("point"):
                        points.append([float(pt.get('x')), float(pt.get('y'))])
                    if len(points) > 1:
                        polylines.append(np.array(points, dtype=np.int32))
                        
        return image_name, polylines
    except Exception as e:
        print(f"Erreur RSML {rsml_path.name}: {e}")
        return None, []

def get_rsml_key_from_path(rsml_path):
    """
    Extrait (Experiment, Box) depuis le chemin du RSML.
    Chemin: .../Condition/Experiment/Box/PlantX/RSML/file.rsml
    """
    try:
        box = rsml_path.parents[2].name
        exp = rsml_path.parents[3].name
        return exp, box
    except IndexError:
        return None, None

def main():
    print("--- Étape 1 : Indexation des données RSML ---")
    
    # Dictionnaire : Clé (Experiment, Box, ImageName) -> Valeur [Liste de polylignes]
    rsml_data = defaultdict(list)
    rsml_count = 0

    if RSML_ROOT_DIR.exists():
        for root, dirs, files in os.walk(RSML_ROOT_DIR):
            for file in files:
                if file.endswith(".rsml"):
                    path = Path(root) / file
                    img_name, lines = parse_rsml_geometry(path)
                    
                    if img_name and lines:
                        exp, box = get_rsml_key_from_path(path)
                        if exp and box:
                            # On ajoute les lignes à la clé unique
                            # (toutes plantes confondues s'ajoutent à la même liste)
                            rsml_data[(exp, box, img_name)].extend(lines)
                            rsml_count += 1
    else:
        print(f"Erreur: Dossier RSML introuvable {RSML_ROOT_DIR}")
        return

    print(f"-> {rsml_count} fichiers RSML traités.")
    print(f"-> {len(rsml_data)} images ont des racines tracées.")

    print("\n--- Étape 2 : Traitement des images et génération des masques ---")
    
    processed_count = 0
    empty_masks_count = 0
    
    # On parcourt le dossier IMAGES
    for root, dirs, files in os.walk(IMAGES_ROOT_DIR):
        for file in files:
            if file.endswith(".png") and not file.endswith("_mask.png"):
                img_path = Path(root) / file
                
                # Identifier Expérience et Boite d'après le chemin de l'image
                # Structure image : .../RAW/Graph_inference/Experiment/Box/Image.png
                try:
                    box_name = img_path.parent.name      # ex: "1"
                    exp_name = img_path.parent.parent.name # ex: "rpi6_..."
                    
                    # 1. Charger l'image (pour avoir la taille exacte HxW)
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    h, w = img.shape
                    mask = np.zeros((h, w), dtype=np.uint8) # Fond noir par défaut
                    
                    # 2. Chercher si on a des données RSML pour cette clé
                    key = (exp_name, box_name, file)
                    
                    if key in rsml_data:
                        # Si oui, on dessine les racines en blanc
                        polylines = rsml_data[key]
                        cv2.polylines(mask, polylines, isClosed=False, color=255, thickness=ROOT_THICKNESS)
                    else:
                        # Si non, on garde le masque noir, mais on le compte
                        empty_masks_count += 1

                    # 3. Sauvegarder le masque
                    # Structure sortie : Output/Experiment/Box/Image_mask.png
                    output_subfolder = OUTPUT_MASK_DIR / exp_name / box_name
                    os.makedirs(output_subfolder, exist_ok=True)
                    
                    mask_filename = f"{img_path.stem}_mask.png"
                    output_path = output_subfolder / mask_filename
                    
                    cv2.imwrite(str(output_path), mask)
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        print(f"Généré ({processed_count}) : {exp_name}/{box_name}/{mask_filename}")

                except IndexError:
                    pass

    print("-" * 30)
    print(f"Terminé.")
    print(f"Total masques générés : {processed_count}")
    print(f"Dont masques vides (pas de RSML associé) : {empty_masks_count}")

if __name__ == "__main__":
    main()