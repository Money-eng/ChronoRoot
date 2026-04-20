import os
import csv
import cv2
import numpy as np
import json
import pandas as pd
import copy
import multiprocessing
import subprocess
import shutil
import re
from concurrent.futures import ProcessPoolExecutor

from .fileFunc import createResultFolder, loadPath
from .imageFunc import getCleanSeg, getCleanSke, savePlotImages, saveEmpty
from .graphFunc import createGraph, saveGraph, saveProps
from .trackFunc import graphInit, matchGraphs
from .rsmlFunc import createTree
from .graphPostProcess import trimGraph
from .dataWork import dataWork

cv2.setNumThreads(0)
DEBUG = False
IS_TAR_GZ = False


def getImgName(image, conf):
    return image.replace(conf['Path'], '').replace('/', '')


def process_plant_task(args):
    """
    Cette fonction sera exécutée par chaque cœur CPU.
    Elle déballe les arguments et lance l'analyse.
    """
    conf_base, images, segFiles, seed_pos_rel, roi_bbox, save_path_plant = args

    local_conf = copy.deepcopy(conf_base)
    local_conf['Project'] = save_path_plant

    try:
        os.makedirs(save_path_plant, exist_ok=True)
        ChronoRootAnalyzer(local_conf, images, segFiles, seed_pos_rel, roi_bbox)

        archive_name = f"{save_path_plant}.tar.gz"

        parent_dir = os.path.dirname(save_path_plant)
        folder_name = os.path.basename(save_path_plant)

        subprocess.run(
            ["tar", "-czf", archive_name, "-C", parent_dir, folder_name],
            check=True
        )

        shutil.rmtree(save_path_plant)

        return f"Success & Archived: {archive_name}"

    except subprocess.CalledProcessError as e:
        return f"Error archiving {save_path_plant}: {str(e)}"
    except Exception as e:
        # remove folder if exists
        if os.path.exists(save_path_plant):
            shutil.rmtree(save_path_plant)
        return f"Error processing {save_path_plant}: {str(e)}"


def PrepareAnalyzer(conf):
    """Structrure 
    - imgs path contains folder of images + 1 csv document with metadata
    - seg path contains folder of epochs with segmentation results (file segmentation name = image name)
    - save path is an empty folder
    """
    img_path = conf['Path']
    seg_path = conf['SegPath']
    save_path = conf['Project']
    save_path = os.path.abspath(save_path)

    # img_name = img_path.split('/')[-1]
    seg_path = os.path.abspath(seg_path)  # os.path.join(seg_path, img_name)
    print("Segmentation path:", seg_path)

    os.makedirs(save_path, exist_ok=True)

    csv_path = os.path.join(img_path, 'seed_and_roi.csv')
    try:
        seed_and_roi = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Cannot find the seed and ROI file at {csv_path}. Please check the path and try again.")
        return

    all_entries = os.listdir(seg_path)
    epoch_numbers = set()
    for entry in all_entries:
        match = re.search(r'epoch_(\d+)', entry)
        if match:
            epoch_numbers.add(int(match.group(1)))
    
    sorted_epochs = sorted(list(epoch_numbers))
    print(f"Epochs detected : {sorted_epochs}")

    tasks = []
    skipped_count = 0
    print("Preparing tasks for analysis...")

    for num in sorted_epochs:
        epoch_name = f"epoch_{num}"
        epoch_path = os.path.join(seg_path, epoch_name)
        tar_gz_file = epoch_path + '.tar.gz'
        save_epoch_path = os.path.join(save_path, epoch_name)

        is_empty = not os.path.exists(epoch_path) or not os.listdir(epoch_path)
        
        if is_empty and os.path.exists(tar_gz_file):
            print(f"Extraction de {tar_gz_file}...")
            os.makedirs(epoch_path, exist_ok=True)
            subprocess.run(["tar", "-xzf", tar_gz_file, "-C", epoch_path], check=True)
        elif is_empty and not os.path.exists(tar_gz_file):
            print(f"No segmentation data found for {epoch_name}, skipping...")
            continue

        os.makedirs(save_epoch_path, exist_ok=True)
        img_folders = [f for f in os.listdir(img_path) if os.path.isdir(os.path.join(img_path, f))]

        for img_folder in img_folders:
            folder_num = int(img_folder)
            
            path_v1 = os.path.join(epoch_path, "EnsembleResult", img_folder)
            path_v2 = os.path.join(epoch_path, img_folder)
            
            if os.path.exists(path_v1):
                seg_path_folder = path_v1
            elif os.path.exists(path_v2):
                seg_path_folder = path_v2
            else:
                if os.listdir(epoch_path): 
                    print(f"Warning: No segmentation folder found for image folder {img_folder} in {epoch_name}")
                continue
            
            img_path_folder = os.path.join(img_path, img_folder)
            save_path_folder = os.path.join(save_epoch_path, img_folder)
            box_row = seed_and_roi[seed_and_roi['Box'] == folder_num]
            if box_row.empty: continue

            for _, row in box_row.iterrows():
                plant_name = row['PlantName']
                save_path_plant = os.path.join(save_path_folder, plant_name)
                archive_file = save_path_plant + ".tar.gz"

                if os.path.exists(archive_file) or os.path.exists(os.path.join(save_path_plant, "Results.csv")):
                    skipped_count += 1
                    continue

                ext = "*" + conf["FileExt"]
                images = loadPath(img_path_folder, ext)
                segFiles = loadPath(seg_path_folder, ext)

                if len(images) == 0 or len(segFiles) == 0:
                    if os.path.exists(seg_path_folder):
                        print(f"Info : No segments found in {seg_path_folder} for plant {plant_name}")
                    continue

                seed_pos = (row['seed_x'], row['seed_y'])
                roi_bbox = (row['y_min'], row['y_max'], row['x_min'], row['x_max'])
                seed_pos_rel = [seed_pos[0] - roi_bbox[2], seed_pos[1] - roi_bbox[0]]

                tasks.append((conf, images, segFiles, seed_pos_rel, np.array(roi_bbox), save_path_plant))

    print(f"Summary: {skipped_count} plants skipped. {len(tasks)} new tasks created.")

    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        max_workers = min(int(slurm_cpus), 192)
    else:
        max_workers = 20

    print(f"Configuration : Utilisation de {max_workers} travailleurs.")

    ctx = multiprocessing.get_context('spawn')

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        results = list(executor.map(process_plant_task, tasks))

    print("Traitement terminé.")
    for res in results:
        if "Error" in res:
            print(res)


def ChronoRootAnalyzer(conf, images, segFiles, seed, bbox):
    lim = conf['Limit']

    if lim != 0:
        images = images[:lim]
        segFiles = segFiles[:lim]

    originalSeed = seed.copy()

    # # plot box on last image for verification + seed point
    # if DEBUG:
    #     import matplotlib.pyplot as plt
    #     last_image = cv2.imread(images[-1])
    #     boxed_image = last_image[bbox[0]:bbox[1], bbox[2]:bbox[3]].copy()
    #     plt.imshow(cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB))
    #     plt.scatter(seed[0], seed[1], c='red', s=50, label='Seed Point')
    #     plt.title('ROI with Seed Point')
    #     plt.legend()
    #     plt.show()

    saveFolder, graphsPath, imagePath, rsmlPath = createResultFolder(conf)

    metadata = {}
    metadata['bounding box'] = bbox.tolist()
    metadata['seed'] = seed
    metadata['folder'] = conf['Path']
    metadata['segFolder'] = conf['SegPath']
    metadata['info'] = conf['fileKey']

    metapath = os.path.join(saveFolder, 'metadata.json')

    with open(metapath, 'w') as fp:
        json.dump(metadata, fp)

    start = 0
    N = len(images)
    pfile = os.path.join(saveFolder, "Results.csv")

    with open(pfile, 'w+') as csv_file:
        csv_writer = csv.writer(csv_file)
        row0 = ['FileName', 'TimeStep', 'MainRootLength', 'LateralRootsLength',
                'NumberOfLateralRoots', 'TotalLength',
                'TotalOrganCount', 'ConvexHullArea', 'RootDensity']
        csv_writer.writerow(row0)

        for i in range(0, N):
            # print('TimeStep', i+1, 'of', N) 
            segFile = segFiles[i]
            seg, segFound = getCleanSeg(segFile, bbox, originalSeed, originalSeed)

            original = cv2.imread(images[i])[bbox[0]:bbox[1], bbox[2]:bbox[3]]

            if segFound:
                ske, bnodes, enodes, flag = getCleanSke(seg)
                if flag:
                    start = i
                    break

            image_name = getImgName(images[i], conf)
            saveProps(image_name, i, False, csv_writer, 0)
            if conf['SaveImages']:
                saveEmpty(image_name, imagePath, original, seg)

        print('Growth Begin')

        grafo, seed, ske2 = createGraph(ske.copy(), seed, enodes, bnodes) # create the initial graph from the skeleton of the first valid segmentation, using the seed point and the end nodes and branch nodes of the skeleton. The graph is built by following the neighbors of the seed point in the skeleton, and adding vertices and edges to the graph accordingly. The function returns the graph, the updated seed point (which is now a vertex in the graph), and a copy of the skeleton that has been modified during the graph creation process.
        grafo, ske, ske2 = trimGraph(grafo, ske, ske2) # post-process the graph and the skeleton to remove any spurious branches or nodes that may have been added during the graph creation process. This function takes the initial graph, the original skeleton, and the modified skeleton as input, and returns a trimmed version of the graph and the skeletons. The trimming process may involve removing small branches, merging close nodes, or other operations to clean up the graph structure and ensure it accurately represents the root system.
        grafo = graphInit(grafo) # initialize the graph for tracking by setting the appropriate properties for the vertices and edges. This function takes the trimmed graph as input and returns an initialized graph that is ready for tracking across time steps. The initialization process may involve setting vertex types (e.g., main root, lateral root), edge classes (e.g., root segments), and other properties that will be used in the tracking process to match nodes and edges across different time steps.

        image_name = getImgName(images[i], conf)

        gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'], '.xml.gz'))
        saveGraph(grafo, gPath)

        rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2) # create an RSML tree structure from the graph and the skeletons. This function takes the configuration, the current time step index, the list of images, the current graph, and the skeletons as input, and returns an RSML tree object that represents the root system structure at the current time step. The function may involve traversing the graph to identify root segments, classifying them as main root or lateral roots based on their properties, and constructing a hierarchical tree structure that can be saved in RSML format for further analysis or visualization.

        rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'], '.rsml'))
        rsmlTree.write(open(rsml, 'w'), encoding='unicode')

        saveProps(image_name, i, grafo, csv_writer, numberLR)

        original = cv2.imread(images[i])[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        if conf['SaveImages']:
            savePlotImages(image_name, imagePath, original, seg, grafo, ske2)

        segErrorFlag = False  # Previous time-step error
        trackCount = 0

        for i in range(start + 1, N):
            print('TimeStep', i + 1, 'of', N)
            errorFlag_ = False

            segFile = segFiles[i]
            seg, flag1 = getCleanSeg(segFile, bbox, seed.tolist(), originalSeed)

            if flag1:
                ske, bnodes, enodes, flag2 = getCleanSke(seg)
                if not flag2:
                    print(
                        f"Error in the skeletonization at time step {i} for image {images[i]} in segmentation, plant {getImgName(images[i], conf)}")
                    errorFlag_ = True
            else:
                print(
                    f"Error in the segmentation at time step {i} for image {images[i]}, plant {getImgName(images[i], conf)}")
                errorFlag_ = True

            trackError = False

            if not errorFlag_:
                grafo2, seed, ske2_ = createGraph(ske.copy(), seed, enodes, bnodes)
                grafo2, ske_, ske2_ = trimGraph(grafo2, ske.copy(), ske2_)

                if not segErrorFlag:
                    try:
                        grafo = matchGraphs(grafo, grafo2)
                        ske = ske_.copy()
                        ske2 = ske2_.copy()
                    except:
                        print(
                            f"Error on node tracking at time step {i} for image {images[i]}, plant {getImgName(images[i], conf)}")
                        trackError = True
                else:
                    grafo = graphInit(grafo2)
                    ske = ske_.copy()
                    ske2 = ske2_.copy()

            else:
                image_name = getImgName(images[i], conf)
                saveProps(image_name, i, False, csv_writer, 0)
                if conf['SaveImages']:
                    saveEmpty(image_name, imagePath, original, seg)

            segErrorFlag = errorFlag_

            if not segErrorFlag and not trackError:
                gPath = os.path.join(graphsPath, image_name.replace(conf['FileExt'], '.xml.gz'))
                saveGraph(grafo, gPath)

                seedrsml = None
                v = grafo[0].get_vertices()
                for k in v:
                    if grafo[4][k] == "Ini":
                        seedrsml = grafo[1][k]
                        seedrsml = np.array(seed, dtype='int')

                if seedrsml is None:
                    trackError = True
                    image_name = images[i].replace(conf['Path'], '').replace('/', '')
                    saveProps(image_name, i, False, csv_writer, 0)
                    if conf['SaveImages']:
                        saveEmpty(image_name, imagePath, original, seg)
                else:
                    rsmlTree, numberLR = createTree(conf, i, images, grafo, ske, ske2)
                    rsml = os.path.join(rsmlPath, image_name.replace(conf['FileExt'], '.rsml'))
                    rsmlTree.write(open(rsml, 'w'), encoding='unicode')

                    image_name = getImgName(images[i], conf)
                    saveProps(image_name, i, grafo, csv_writer, numberLR)

                    original = cv2.imread(images[i])[bbox[0]:bbox[1], bbox[2]:bbox[3]]
                    if conf['SaveImages']:
                        savePlotImages(image_name, imagePath, original, seg, grafo, ske2)

            if trackError and trackCount > 5:
                print('Analysis ended early at timestep', i, 'of', N)
                break
            elif trackError:
                trackCount += 1
            else:
                trackCount = 0

    dataWork(conf, pfile, saveFolder)
