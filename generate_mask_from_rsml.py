import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# ================= CONFIGURATION =================
# Chemin vers le dossier racine contenant l'arborescence RSML (ContLight, LongDay...)
RSML_ROOT_DIR = "./DataTest/ChronoRoot/Graphs" 

# Chemin vers le dossier racine contenant les Images brutes (.png)
# Le script cherchera les images en suivant la structure : Condition/Experiment/Boite/Image.png
IMAGES_ROOT_DIR = "./DataTest/ChronoRoot/Images" 

# Dossier où sauvegarder les masques générés
OUTPUT_MASK_DIR = "./DataTest/ChronoRoot/Masks"

# Diamètre de la racine en pixels pour le masque
ROOT_THICKNESS = 5 
# =================================================

def parse_rsml_geometry(rsml_path):
    """
    Simule le comportement de openalea.rsml pour extraire
    le nom de l'image cible et les polylignes des racines.
    """
    tree = ET.parse(rsml_path)
    root = tree.getroot()
    
    # 1. Récupérer le nom de l'image associée
    # Le chemin dans le XML est <metadata><image><name>
    image_name_node = root.find(".//metadata/image/name")
    if image_name_node is None:
        return None, []
    
    image_name = image_name_node.text
    
    # 2. Récupérer toutes les polylignes (racines)
    # Structure: <scene><plant><root><geometry><polyline><point>
    polylines = []
    
    # On itère sur toutes les plantes et toutes les racines dans le fichier
    for plant in root.findall(".//scene/plant"):
        for root_entity in plant.findall(".//root"):
            polyline_node = root_entity.find(".//geometry/polyline")
            if polyline_node is not None:
                points = []
                for pt in polyline_node.findall("point"):
                    x = float(pt.get('x'))
                    y = float(pt.get('y'))
                    points.append([x, y])
                
                if len(points) > 1:
                    polylines.append(np.array(points, dtype=np.int32))
                    
    return image_name, polylines

def find_image_path(rsml_full_path, image_filename):
    """
    Reconstruit le chemin de l'image basé sur la position du RSML.
    Logique:
    RSML: .../Condition/Experience/Boite/PlantX/RSML/file.rsml
    Image: .../Condition/Experience/Boite/image_filename
    """
    path_obj = Path(rsml_full_path)
    
    # On remonte de 3 niveaux pour sortir de RSML/PlantX/
    # Parents: [0]=RSML, [1]=PlantX, [2]=Boite (ex: 1, 2, 3, 4)
    try:
        box_dir_name = path_obj.parents[2].name     # ex: "1"
        exp_dir_name = path_obj.parents[3].name     # ex: "rpi14_..."
        cond_dir_name = path_obj.parents[4].name    # ex: "ContLight"
        
        # On construit le chemin théorique de l'image
        # Note: Ajustez cette concaténation selon la structure exacte de votre dossier Images
        image_full_path = Path(IMAGES_ROOT_DIR) / cond_dir_name / exp_dir_name / box_dir_name / image_filename
        
        return image_full_path, (cond_dir_name, exp_dir_name, box_dir_name)
    except IndexError:
        print(f"Erreur de structure pour : {rsml_full_path}")
        return None, None

def main():
    print("--- Démarrage de la génération des masques ---")
    
    # Dictionnaire pour regrouper les racines par image cible
    # Clé : Chemin absolu de l'image
    # Valeur : Liste de toutes les polylignes (venant potentiellement de plusieurs RSML/Plantes)
    tasks = defaultdict(list)
    
    # 1. Parcours et Lecture des RSML
    rsml_count = 0
    for root, dirs, files in os.walk(RSML_ROOT_DIR):
        for file in files:
            if file.endswith(".rsml"):
                rsml_path = os.path.join(root, file)
                
                img_name, polylines = parse_rsml_geometry(rsml_path)
                
                if img_name and polylines:
                    # Trouver où est l'image correspondante
                    img_path, structure_info = find_image_path(rsml_path, img_name)
                    
                    if img_path:
                        # On stocke l'info nécessaire pour dessiner plus tard
                        tasks[img_path].extend(polylines)
                        rsml_count += 1

    print(f"Analyse terminée : {rsml_count} fichiers RSML traités.")
    print(f"Nombre d'images uniques à générer : {len(tasks)}")
    
    # 2. Génération des Masques
    for img_path_obj, all_polylines in tasks.items():
        # Convertir en string pour OpenCV
        img_path_str = str(img_path_obj)
        
        # Vérifier si l'image existe pour avoir ses dimensions
        if not os.path.exists(img_path_str):
            print(f"[Attention] Image source introuvable : {img_path_str}. Masque ignoré.")
            # Optionnel : On pourrait créer un masque de taille par défaut (ex: 2000x2000)
            continue
            
        # Charger l'image juste pour avoir les dimensions (H, W)
        # On lit en "unchanged" ou "grayscale" pour aller vite
        original_img = cv2.imread(img_path_str, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            print(f"[Erreur] Impossible de lire l'image : {img_path_str}")
            continue
            
        h, w = original_img.shape[:2]
        
        # Créer un masque noir (0)
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Dessiner les racines en blanc (255)
        # polylines est une liste de tableaux numpy de points
        cv2.polylines(mask, all_polylines, isClosed=False, color=255, thickness=ROOT_THICKNESS)
        
        # 3. Sauvegarde
        # On recrée la structure de dossier dans le dossier de sortie
        # img_path_obj ressemble à .../Images/ContLight/rpi.../1/image.png
        # On veut .../Masks/ContLight/rpi.../1/image_mask.png
        
        # Calcul du chemin relatif par rapport au dossier racine des images
        try:
            relative_path = img_path_obj.relative_to(IMAGES_ROOT_DIR)
        except ValueError:
            # Si le chemin ne correspond pas, on essaie de garder la structure dossier parent
            relative_path = Path(img_path_obj.parent.name) / img_path_obj.name

        save_path = Path(OUTPUT_MASK_DIR) / relative_path.parent / f"{img_path_obj.stem}_mask.png"
        
        # Créer les dossiers parents si inexistants
        os.makedirs(save_path.parent, exist_ok=True)
        
        cv2.imwrite(str(save_path), mask)
        print(f"Généré : {save_path.name}")

    print("--- Terminé ---")

if __name__ == "__main__":
    main()