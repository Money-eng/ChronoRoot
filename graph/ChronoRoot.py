""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion (Modified Wrapper for Multi-Root Analysis)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
import csv
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.ndimage import label, find_objects

# Imports originaux de ChronoRoot
from .fileFunc import createResultFolder, loadPath, getROIandSeed
from .imageFunc import getCleanSeg, getCleanSke, savePlotImages, saveEmpty
from .graphFunc import createGraph, saveGraph, saveProps
from .trackFunc import graphInit, matchGraphs
from .rsmlFunc import createTree
from .graphPostProcess import trimGraph
from .dataWork import dataWork

# --- Configuration ---
DEBUG_VISUALIZATION = True  # Mettre à True pour voir les boîtes et seeds avant analyse

def getImgName(image, conf):
    return image.replace(conf['Path'], '').replace('/', '')

def find_stable_seed(seg_files, bbox, obj_mask_last, first_seg_full):
    """
    Cherche un point de départ (seed) stable en calculant l'intersection
    temporelle de la racine à travers toutes les images.
    """
    y_min, y_max, x_min, x_max = bbox
    
    # 1. Initialisation avec le masque de l'objet sur la dernière image
    temporal_intersection = obj_mask_last.copy()
    
    # 2. Intersection temporelle stricte (remonter le temps ou parcourir tout)
    for segFile in seg_files:
        img_t = cv2.imread(segFile, cv2.IMREAD_GRAYSCALE)
        if img_t is None: continue # Sécurité image corrompue

        crop_t = img_t[y_min:y_max, x_min:x_max]
        _, bin_crop_t = cv2.threshold(crop_t, 127, 255, cv2.THRESH_BINARY)
        
        # Intersection logique
        temporal_intersection = cv2.bitwise_and(temporal_intersection, bin_crop_t)
        
        # Optimisation : Si intersection vide, inutile de continuer
        if not np.any(temporal_intersection):
            break
            
    # 3. Extraction du seed (point le plus haut)
    ys, xs = np.where(temporal_intersection == 255)
    
    if len(ys) > 0:
        # On prend le point le plus haut (y min)
        idx = ys.argmin()
        return int(ys[idx]), int(xs[idx])
    
    # 4. Fallback : Intersection First & Last seulement si l'intersection totale échoue
    # (Cas où la racine a bougé un peu mais le début reste fixe)
    sub_first = first_seg_full[y_min:y_max, x_min:x_max]
    ys_fb, xs_fb = np.where((sub_first == 255) & (obj_mask_last == 255))
    
    if len(ys_fb) > 0:
        idx = ys_fb.argmin()
        return int(ys_fb[idx]), int(xs_fb[idx])
        
    return 0, 0 # Échec total

def analyze_single_component(conf, images, segFiles, bbox, local_seed, root_id):
    """
    Exécute le pipeline ChronoRoot complet pour UNE seule racine identifiée.
    """
    print(f"\n--- Processing Root Component #{root_id} ---")
    
    # Conversion seed local (dans bbox) -> global (dans image)
    # Le code original semble parfois utiliser seed relative ou absolue selon les fonctions.
    # getCleanSeg prend généralement (bbox, seed). Si getCleanSeg crop l'image, 
    # seed doit être relative. Si elle utilise l'image entière, seed doit être absolue.
    # Dans le doute, basons-nous sur l'usage standard : souvent on passe le seed global.
    global_seed = [int(local_seed[0]), int(local_seed[1])]
    
    # IMPORTANT : Cloner la conf pour ne pas écraser les résultats des autres racines
    # On modifie le 'fileKey' ou le dossier de sortie pour séparer les résultats
    current_conf = conf.copy()
    original_key = current_conf.get('fileKey', 'result')
    current_conf['fileKey'] = f"{original_key}_root_{root_id}"
    
    # Création des dossiers pour CETTE racine
    saveFolder, graphsPath, imagePath, rsmlPath = createResultFolder(current_conf)
    
    # Metadata
    metadata = {
        'bounding box': bbox,
        'seed': global_seed,
        'folder': current_conf['Path'],
        'segFolder': current_conf['SegPath'],
        'info': current_conf['fileKey'],
        'root_id': root_id
    }
    
    with open(os.path.join(saveFolder, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp)

    # --- Début du pipeline original ---
    N = len(images)
    start = 0
    pfile = os.path.join(saveFolder, "Results.csv")
    
    # Variables d'état
    ske = None
    ske2 = None
    grafo = None
    
    with open(pfile, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['FileName', 'TimeStep','MainRootLength','LateralRootsLength','NumberOfLateralRoots','TotalLength'])
        
        # Phase 1: Trouver la première segmentation valide
        for i in range(N):
            print(f"Root {root_id} - Init Step {i+1}/{N}")
            segFile = segFiles[i]
            # Note: getCleanSeg utilise global_seed ici selon la logique usuelle
            seg, segFound = getCleanSeg(segFile, np.array(bbox), global_seed, global_seed)
            
            # Chargement Crop Image Originale
            full_img = cv2.imread(images[i])
            if full_img is None: continue
            original = full_img[bbox[0]:bbox[1], bbox[2]:bbox[3]]
            
            if segFound:
                ske, bnodes, enodes, flag = getCleanSke(seg)
                if flag:
                    start = i
                    break
            
            # Sauvegarde frames vides
            image_name = getImgName(images[i], current_conf)
            saveProps(image_name, i, False, csv_writer, 0)
            saveEmpty(image_name, imagePath, original, seg)
            
        print(f"Root {root_id} - Growth Begin at step {start}")
        
        # Initialisation du Graphe
        grafo, seed_graph, ske2 = createGraph(ske.copy(), global_seed, enodes, bnodes)
        if hasattr(seed_graph, '__iter__') or isinstance(seed_graph, np.ndarray):
             # Écrase les types numpy (int64) par des int Python
            seed_graph = [int(seed_graph[0]), int(seed_graph[1])]
        else:
            # Fallback si format inattendu
            seed_graph = [int(global_seed[0]), int(global_seed[1])]
        grafo, ske, ske2 = trimGraph(grafo, ske, ske2)
        grafo = graphInit(grafo)
        
        # Sauvegarde premier graphe
        image_name = getImgName(images[start], current_conf)
        gPath = os.path.join(graphsPath, image_name.replace(current_conf['FileExt'],'.xml.gz'))
        saveGraph(grafo, gPath)
        
        rsmlTree, numberLR = createTree(current_conf, start, images, grafo, ske, ske2)
        rsmlTree.write(open(os.path.join(rsmlPath, image_name.replace(current_conf['FileExt'],'.rsml')), 'w'), encoding='unicode')
        
        saveProps(image_name, start, grafo, csv_writer, numberLR)
        savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
        
        # Phase 2: Tracking temporel
        segErrorFlag = False
        trackCount = 0
        
        for i in range(start + 1, N):
            print(f"Root {root_id} - Tracking Step {i+1}/{N}")
            errorFlag_ = False
            trackError = False
            
            segFile = segFiles[i]
            seg, flag1 = getCleanSeg(segFile, np.array(bbox), seed_graph, global_seed)
            
            # Re-load original for display
            full_img = cv2.imread(images[i])
            original = full_img[bbox[0]:bbox[1], bbox[2]:bbox[3]] if full_img is not None else np.zeros_like(seg)

            if flag1:
                ske_curr, bnodes, enodes, flag2 = getCleanSke(seg)
                if not flag2:
                    print("Error in skeleton")
                    errorFlag_ = True
            else:
                print("Error in segmentation")
                errorFlag_ = True
            
            if not errorFlag_:
                grafo2, _, ske2_ = createGraph(ske_curr.copy(), seed_graph, enodes, bnodes)
                grafo2, ske_, ske2_ = trimGraph(grafo2, ske_curr.copy(), ske2_)
                
                if not segErrorFlag:
                    try:
                        grafo = matchGraphs(grafo, grafo2, seg=seg)
                        ske = ske_.copy()
                        ske2 = ske2_.copy()
                    except Exception as e:
                        print(f"Error on node tracking: {e}")
                        trackError = True
                else:
                    # Reset tracking if previous frame was bad
                    grafo = graphInit(grafo2)
                    ske = ske_.copy()
                    ske2 = ske2_.copy()
            else:
                image_name = getImgName(images[i], current_conf)
                saveProps(image_name, i, False, csv_writer, 0)
                saveEmpty(image_name, imagePath, original, seg)
                
            segErrorFlag = errorFlag_
            
            if not segErrorFlag and not trackError:
                image_name = getImgName(images[i], current_conf)
                gPath = os.path.join(graphsPath, image_name.replace(current_conf['FileExt'],'.xml.gz'))
                saveGraph(grafo, gPath)
                
                # Vérification seed dans graphe
                seedrsml = None
                v = grafo[0].get_vertices()
                for k in v:
                    if grafo[4][k] == "Ini":
                        seedrsml = grafo[1][k] # Just check existence
                
                if seedrsml is None:
                    trackError = True
                    saveProps(image_name, i, False, csv_writer, 0)
                    saveEmpty(image_name, imagePath, original, seg)
                else:
                    rsmlTree, numberLR = createTree(current_conf, i, images, grafo, ske, ske2)
                    rsmlTree.write(open(os.path.join(rsmlPath, image_name.replace(current_conf['FileExt'],'.rsml')), 'w'), encoding='unicode')
                    
                    saveProps(image_name, i, grafo, csv_writer, numberLR)
                    savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
            
            # Gestion d'arrêt prématuré si trop d'erreurs
            if trackError and trackCount > 5:
                print(f"Analysis ended early at timestep {i}")
                break
            elif trackError:
                trackCount += 1
            else:
                trackCount = 0
                
    # Post-traitement des données CSV
    dataWork(current_conf, pfile, saveFolder)


def ChronoRootAnalyzer(conf):
    # 1. Chargement des fichiers
    ext = "*" + conf["FileExt"]
    all_files_img = loadPath(conf['Path'], ext)
    images = [f for f in all_files_img] # Filtre basique
    
    all_files_seg = loadPath(conf['SegPath'], ext)
    segFiles = [f for f in all_files_seg]   # Filtre basique (selon ton code)
    
    # Application de la limite
    lim = conf.get('Limit', 0)
    if lim != 0:
        images = images[:lim]
        segFiles = segFiles[:lim]

    if not segFiles:
        print("No segmentation files found.")
        return

    # 2. Détection des composantes (Racines multiples)
    print("Detecting root systems in the last frame...")
    
    lastSegFile = segFiles[-1]
    firstSegFile = segFiles[0]
    
    # Chargement images de référence (binaire)
    segLast = cv2.imread(lastSegFile, cv2.IMREAD_GRAYSCALE)
    segLast = cv2.threshold(segLast, 127, 255, cv2.THRESH_BINARY)[1]
    
    segFirst = cv2.imread(firstSegFile, cv2.IMREAD_GRAYSCALE)
    segFirst = cv2.threshold(segFirst, 127, 255, cv2.THRESH_BINARY)[1]
    
    # Étiquetage des composantes connexes
    labeled_array, num_features = label(segLast)
    objects = find_objects(labeled_array)
    
    roots_to_process = []
    
    for i, obj in enumerate(objects):
        if obj is None: continue
        
        # Extraction BBox
        y_min, y_max = int(obj[0].start), int(obj[0].stop)
        x_min, x_max = int(obj[1].start), int(obj[1].stop)
        bbox = (y_min, y_max, x_min, x_max)
        
        # Création masque local pour isoler cet objet spécifique
        # (Pour ne pas calculer l'intersection avec la racine voisine)
        sub_mask = (labeled_array[y_min:y_max, x_min:x_max] == (i + 1)).astype(np.uint8) * 255
        
        # Recherche du Seed Stable
        local_seed_y, local_seed_x = find_stable_seed(segFiles, bbox, sub_mask, segFirst)
        
        # Si le seed est (0,0), c'est souvent une erreur ou un artefact, on peut filtrer ici si besoin
        roots_to_process.append({
            'id': i + 1,
            'bbox': bbox,
            'local_seed': (local_seed_y, local_seed_x)
        })

    print(f"Found {len(roots_to_process)} separate root systems.")

    # 3. Visualisation (Optionnelle)
    if DEBUG_VISUALIZATION:
        plt.figure(figsize=(10, 10))
        plt.title("Detected Roots & Stable Seeds (Close window to continue)")
        plt.imshow(segLast, cmap='gray')
        plt.imshow(segFirst, cmap='jet', alpha=0.3)
        
        for root in roots_to_process:
            bb = root['bbox'] # ymin, ymax, xmin, xmax
            ls = root['local_seed']
            
            # Rectangle
            rect = plt.Rectangle((bb[2], bb[0]), bb[3]-bb[2], bb[1]-bb[0],
                                 edgecolor='red', facecolor='none', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Seed (Global coordinate for plot)
            plt.scatter(ls[1] + bb[2], ls[0] + bb[0], c='yellow', marker='x', s=100)
            plt.text(bb[2], bb[0]-5, f"ID {root['id']}", color='red', fontsize=12)
            
        plt.show()

    # 4. Lancement de l'analyse pour chaque racine
    for root in roots_to_process:
        try:
            analyze_single_component(
                conf, 
                images, 
                segFiles, 
                root['bbox'], 
                root['local_seed'], 
                root['id']
            )
        except Exception as e:
            print(f"CRITICAL ERROR processing root {root['id']}: {e}")
            import traceback
            traceback.print_exc()

    print("All roots processed.")