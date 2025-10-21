""" 
ChronoRoot: High-throughput phenotyping by deep learning reveals novel temporal parameters of plant root system architecture
Copyright (C) 2020 Nicolás Gaggion

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import csv
import cv2
import numpy as np
import json
from pathlib import Path

from .fileFunc import createResultFolder, loadPath, getROIandSeed
from .imageFunc import getCleanSeg, getCleanSke, savePlotImages, saveEmpty
from .graphFunc import createGraph, saveGraph, saveProps
from .trackFunc import graphInit, matchGraphs
from .rsmlFunc import createTree
from .graphPostProcess import trimGraph
from .dataWork import dataWork

plant_number = 0

def getImgName(image, conf, index):
    global plant_number
    return conf['Project'] + '/' + conf['fileKey'] +  '_I' + str(index) + '_P' + str(plant_number) + conf['FileExt']

def _to_jsonable(obj):
    """Recursively convert common non-JSON types (numpy, Path, sets) to JSON-safe forms."""
    # numpy scalar types
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # paths
    if isinstance(obj, Path):
        return str(obj)
    # sets/tuples
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, tuple):
        return list(obj)
    # mappings and sequences
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list,)):
        return [_to_jsonable(v) for v in obj]
    # fallback
    return obj

def automatic_seed_from_segmentation(seg: np.ndarray, rsml_path: str, time_step: int):
    """
    map_label_to_seed = {}
    map_label_to_bounding_box = {}
    for plant in plants:
        roots = root_vertices(plant)
        primary_root = roots[0]  # 1 primary root per plant
        geom = plant.properties()['geometry'][primary_root]
        for x, y in geom:
            x, y = int(round(x)), int(round(y))
            label_at_pos = ccs[0][y, x]
            if label_at_pos > 0 and seg[y, x] > 0:
                if label_at_pos in map_label_to_seed:
                    print(f"Warning: multiple seeds for label {label_at_pos}, keeping none")
                    # remove previous seed
                    del map_label_to_seed[label_at_pos]
                    break
                map_label_to_seed[label_at_pos] = (x, y)
                # find bounding box of the connected component
                ys, xs = np.where(ccs[0] == label_at_pos)
                min_x, max_x = xs.min(), xs.max()
                min_y, max_y = ys.min(), ys.max()
                map_label_to_bounding_box[label_at_pos] = (min_y, max_y, min_x, max_x)
                break
    """
    from rsml import rsml2mtg
    from scipy.ndimage import label
    from utils.mtg_operations import extract_mtg_at_time_t, extract_plant_sub_mtg
    from rsml.misc import plant_vertices, root_vertices

    ccs = label(seg)
    mtg_gt = rsml2mtg(rsml_path)
    
    mtg_gt_t = extract_mtg_at_time_t(mtg_gt, time_step)
    plants = [extract_plant_sub_mtg(mtg_gt_t, r) for r in plant_vertices(mtg_gt_t)]
    
    # for each plant, compute its bounding box
    # and check if it intersects with any other plant
    map_label_to_bounding_box = {}
    map_label_to_seed = {}
    map_label_to_plant = {}
    for plant in plants:
        xs = []
        ys = []
        lab = - 1
        map_label_to_plant = {lab: None}
        for r in root_vertices(plant):
            geom = plant.properties()['geometry'][r]
            xs.extend([int(round(x)) for x, y in geom])
            ys.extend([int(round(y)) for x, y in geom])
            if map_label_to_plant[lab] != plant:
                for x, y in geom:
                    x, y = int(round(x)), int(round(y))
                    label_at_pos = ccs[0][y, x]
                    if label_at_pos > 0 and seg[y, x] > 0:
                        map_label_to_seed[label_at_pos] = (x, y)
                        lab = label_at_pos
                        map_label_to_plant[label_at_pos] = plant
                        break
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        map_label_to_bounding_box[lab] = (min_y, max_y, min_x, max_x)
    
    # # plot label, seed, bounding box on seg
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    # from matplotlib.cm import get_cmap
    # cmap = get_cmap('tab20')
    # fig, ax = plt.subplots(1)
    # ax.imshow(seg, cmap='gray')
    # for i, (lab, seed) in enumerate(map_label_to_seed.items()):
    #     color = cmap(i % 20)
    #     ax.plot(seed[0], seed[1], 'o', color=color, markersize=10)
    #     if lab in map_label_to_bounding_box:
    #         bbox = map_label_to_bounding_box[lab]
    #         rect = patches.Rectangle((bbox[2], bbox[0]), bbox[3]-bbox[2], bbox[1]-bbox[0], linewidth=2, edgecolor=color, facecolor='none')
    #         ax.add_patch(rect)
    #         ax.text(bbox[2], bbox[0]-5, str(lab), color=color, fontsize=12, weight='bold')
    # plt.show()
    
    return map_label_to_seed, map_label_to_bounding_box



def ChronoRootAnalyzer(conf: dict, images: list, segFiles: list, rsml_path: str):
    global plant_number
    plant_number = 0
    # Select connected component (assuming roots do not cross) and select seed point in the roi 
    label_2_seed, label_2_bbox = automatic_seed_from_segmentation(segFiles[-1], rsml_path, -1)
    for label in label_2_seed.keys():
        plant_number += 1
        
        seed = label_2_seed[label]
        bbox = label_2_bbox[label]
        seed = list(seed)
        originalSeed = seed.copy()
        
        saveFolder, graphsPath, imagePath, rsmlPath = createResultFolder(conf)
        
        metadata = {}
        metadata['bounding box'] = list(bbox)
        metadata['seed'] = seed
        metadata['folder'] = conf['Path']
        #metadata['segFolder'] = conf['SegPath']
        metadata['info'] = conf['fileKey']

        print(metadata)
        metapath = os.path.join(saveFolder, 'metadata.json')

        with open(metapath, 'w') as fp:
            json.dump(_to_jsonable(metadata), fp)

        start = 0
        N = len(images)
        pfile = os.path.join(saveFolder, "Results.csv") # For CSV Saver
        conf['captureTime'] = conf['captureTimes'][0]
        
        with open(pfile, 'w+') as csv_file:
            image_name = getImgName(images[0], conf, 0)
            csv_writer = csv.writer(csv_file)
            row0 = ['FileName', 'TimeStep','MainRootLength','LateralRootsLength','NumberOfLateralRoots','TotalLength']
            csv_writer.writerow(row0)

            ### First, it begins by obtaining the first segmentation
            for i in range(0, N+1):
                print('TimeStep', i+1, 'of', N)
                segFile = segFiles[i]
                seg, segFound = getCleanSeg(segFile, bbox, originalSeed, originalSeed)
                
                original = images[i][bbox[0]:bbox[1],bbox[2]:bbox[3]] # cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
                
                if segFound:
                    ske, bnodes, enodes, flag = getCleanSke(seg) # Skeleton, branch nodes, end nodes and flag
                    if flag:
                        start = i
                        break
                
                image_name = getImgName(images[i], conf, i)
                saveProps(image_name, i, False, csv_writer, i) # Save empty properties
                saveEmpty(image_name, imagePath, original, seg) # Save empty images
            
            print('Growth Begin')
            
            grafo, seed, ske2 = createGraph(ske.copy(), seed, enodes, bnodes) # Create networkx graph from skeleton
            grafo, ske, ske2 = trimGraph(grafo, ske, ske2)
            grafo = graphInit(grafo)
            
            image_name = getImgName(images[i], conf, i)
            gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'],'.xml.gz'))
            saveGraph(grafo, gPath)
            
            rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2)
            
            rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'],'.rsml'))
            rsmlTree.write(open(rsml, 'w'), encoding='unicode')        

            saveProps(image_name, i, grafo, csv_writer, numberLR)

            original = images[i][bbox[0]:bbox[1],bbox[2]:bbox[3]] # cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
            
            segErrorFlag = False #Previous time-step error
            trackCount = 0
            
            for i in range(start+1, N):
                image_name = getImgName(images[i], conf, i)
                print('TimeStep', i+1, 'of', N)
                conf['captureTime'] = conf['captureTimes'][i]
                errorFlag_ = False
                
                segFile = segFiles[i]
                seg, flag1 = getCleanSeg(segFile, bbox, seed.tolist(), originalSeed)
                
                if flag1:
                    ske, bnodes, enodes, flag2 = getCleanSke(seg)
                    if not flag2:
                        print("Error in the skeleton")
                        errorFlag_ = True
                else:
                    print("Error in the segmentation")
                    errorFlag_ = True
                
                trackError = False
            
                if not errorFlag_:               
                    grafo2, seed, ske2_ = createGraph(ske.copy(), seed, enodes, bnodes)
                    grafo2, ske_, ske2_ = trimGraph(grafo2, ske.copy(), ske2_)
                    
                    if not segErrorFlag:
                        try:
                            grafo = matchGraphs(grafo, grafo2) # good for buinding 2D+t rsmls
                            ske =  ske_.copy()
                            ske2 = ske2_.copy()
                        except:
                            print("Error on node tracking")
                            trackError = True
                    else:
                        grafo = graphInit(grafo2)
                        ske =  ske_.copy()
                        ske2 = ske2_.copy()
                        
                else:
                    image_name = getImgName(images[i], conf, i)
                    saveProps(image_name, i, False, csv_writer, i)
                    saveEmpty(image_name, imagePath, original, seg)
                
                segErrorFlag = errorFlag_
                            
                if not segErrorFlag and not trackError:           
                    gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'],'.xml.gz'))
                    saveGraph(grafo, gPath)
            
                    seedrsml = None
                    v = grafo[0].get_vertices()
                    for k in v:
                        if grafo[4][k] == "Ini":
                            seedrsml = grafo[1][k]
                            seedrsml = np.array(seed, dtype='int')
                    
                    if seedrsml is None:
                        trackError = True
                        image_name = getImgName(images[i], conf, i)
                        saveProps(image_name, i, False, csv_writer, i)
                        saveEmpty(image_name, imagePath, original, seg)
                    else:
                        rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2)
                        rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'],'.rsml'))
                        rsmlTree.write(open(rsml, 'w'), encoding='unicode')        
                        image_name = getImgName(images[i], conf, i)
                        saveProps(image_name, i, grafo, csv_writer, numberLR)

                        original = images[i][bbox[0]:bbox[1],bbox[2]:bbox[3]]
                        savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
            
                if trackError and trackCount > 5:
                    print('Analysis ended early at timestep', i, 'of', N)
                    break
                elif trackError:
                    trackCount += 1
                else:
                    trackCount = 0
        
        try:
            dataWork(conf, pfile, saveFolder)
        except:
            print("Error in dataWork")
            pass

def ChronoRootAnalyzerOLD(conf):
    ext = "*" + conf["FileExt"]
    all_files = loadPath(conf['Path'], ext) 
    print(all_files)
    images = [file for file in all_files if 'mask' not in file]
       
    ext = "*" + conf["FileExt"]
    all_files = loadPath(conf['SegPath'], ext) 
    print(all_files)
    segFiles = [file for file in all_files if 'mask' in file] # look if segmentation files exist by checking 'mask' in the name
    
    lim = conf['Limit'] 
    
    print('Number of images found:', len(images))
    print('Number of segmentations found:', len(segFiles))
    
    if lim!=0:
        images = images[:lim]
        segFiles = segFiles[:lim]

    # Select connected component (assuming roots do not cross) and select seed point in the roi 
    bbox, seed = getROIandSeed(conf, images, segFiles) # bounding box and seed point
    seed = list(seed[0])
    originalSeed = seed.copy()
    
    saveFolder, graphsPath, imagePath, rsmlPath = createResultFolder(conf)
    
    metadata = {}
    metadata['bounding box'] = bbox.tolist()
    metadata['seed'] = seed
    metadata['folder'] = conf['Path']
    metadata['segFolder'] = conf['SegPath']
    metadata['info'] = conf['fileKey']

    print(metadata)
    metapath = os.path.join(saveFolder, 'metadata.json')

    with open(metapath, 'w') as fp:
        json.dump(_to_jsonable(metadata), fp)

    start = 0
    N = len(images)
    pfile = os.path.join(saveFolder, "Results.csv") # For CSV Saver
    
    with open(pfile, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        row0 = ['FileName', 'TimeStep','MainRootLength','LateralRootsLength','NumberOfLateralRoots','TotalLength']
        csv_writer.writerow(row0)
        
        ### First, it begins by obtaining the first segmentation
        for i in range(0, N):
            print('TimeStep', i+1, 'of', N)
            segFile = segFiles[i]
            seg, segFound = getCleanSeg(segFile, bbox, originalSeed, originalSeed)
            
            original = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
            
            if segFound:
                ske, bnodes, enodes, flag = getCleanSke(seg) # Skeleton, branch nodes, end nodes and flag
                if flag:
                    start = i
                    break
            
            image_name = getImgName(images[i], conf, i)
            saveProps(image_name, i, False, csv_writer, 0) # Save empty properties
            saveEmpty(image_name, imagePath, original, seg) # Save empty images
        
        print('Growth Begin')
        
        grafo, seed, ske2 = createGraph(ske.copy(), seed, enodes, bnodes) # Create networkx graph from skeleton
        grafo, ske, ske2 = trimGraph(grafo, ske, ske2)
        grafo = graphInit(grafo)
        
        image_name = getImgName(images[i], conf, i)
        
        gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'],'.xml.gz'))
        saveGraph(grafo, gPath)
        
        rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2)
        
        rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'],'.rsml'))
        rsmlTree.write(open(rsml, 'w'), encoding='unicode')        
        
        saveProps(image_name, i, grafo, csv_writer, numberLR)
        
        original = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
        savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
        
        segErrorFlag = False #Previous time-step error
        trackCount = 0
        
        for i in range(start+1, N):
            print('TimeStep', i+1, 'of', N)
            errorFlag_ = False
            
            segFile = segFiles[i]
            seg, flag1 = getCleanSeg(segFile, bbox, seed.tolist(), originalSeed)
            
            if flag1:
                ske, bnodes, enodes, flag2 = getCleanSke(seg)
                if not flag2:
                    print("Error in the skeleton")
                    errorFlag_ = True
            else:
                print("Error in the segmentation")
                errorFlag_ = True
            
            trackError = False
        
            if not errorFlag_:               
                grafo2, seed, ske2_ = createGraph(ske.copy(), seed, enodes, bnodes)
                grafo2, ske_, ske2_ = trimGraph(grafo2, ske.copy(), ske2_)
                
                if not segErrorFlag:
                    try:
                        grafo = matchGraphs(grafo, grafo2)
                        ske =  ske_.copy()
                        ske2 = ske2_.copy()
                    except:
                        print("Error on node tracking")
                        trackError = True
                else:
                    grafo = graphInit(grafo2)
                    ske =  ske_.copy()
                    ske2 = ske2_.copy()
                    
            else:
                image_name = getImgName(images[i], conf, i)
                saveProps(image_name, i, False, csv_writer, 0)
                saveEmpty(image_name, imagePath, original, seg)
            
            segErrorFlag = errorFlag_
                        
            if not segErrorFlag and not trackError:           
                gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'],'.xml.gz'))
                saveGraph(grafo, gPath)
        
                seedrsml = None
                v = grafo[0].get_vertices()
                for k in v:
                    if grafo[4][k] == "Ini":
                        seedrsml = grafo[1][k]
                        seedrsml = np.array(seed, dtype='int')
                
                if seedrsml is None:
                    trackError = True
                    image_name = images[i].replace(conf['Path'],'').replace('/','')
                    saveProps(image_name, i, False, csv_writer, 0)
                    saveEmpty(image_name, imagePath, original, seg)
                else:
                    rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2)
                    rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'],'.rsml'))
                    rsmlTree.write(open(rsml, 'w'), encoding='unicode')        
        
                    image_name = getImgName(images[i], conf, i)
                    saveProps(image_name, i, grafo, csv_writer, numberLR)
                    
                    original = cv2.imread(images[i])[bbox[0]:bbox[1],bbox[2]:bbox[3]]
                    savePlotImages(image_name, imagePath, original, seg, grafo, ske2)
        
            if trackError and trackCount > 5:
                print('Analysis ended early at timestep', i, 'of', N)
                break
            elif trackError:
                trackCount += 1
            else:
                trackCount = 0
    
    dataWork(conf, pfile, saveFolder)
  