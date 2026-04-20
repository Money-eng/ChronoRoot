import os
import glob
import re
import json
import pandas as pd
import numpy as np
import sys
import cv2
import xml.etree.ElementTree as ET
from datetime import datetime
from openalea.rsml import rsml2mtg

try:
    from Metric_4_GT import get_measures
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Metric_4_GT import get_measures

try:
    from graph.qr import qr_detect, get_pixel_size
except ImportError:
    print("Module 'graph.qr' not found. QR code detection will be disabled.")
    from pyzbar import pyzbar # below is a copy

    def adjust_gamma(image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def qr_detect(inputImage):
        img = cv2.imread(inputImage, 0)
        if img is None: return None
        h, w = img.shape
        roi = img[0:int(h//2), int(w//4):int(3*w//4)]
        for gamma in [1.0, 1.2, 1.5, 0.8]:
            adjust = adjust_gamma(roi, gamma)
            data = pyzbar.decode(adjust)
            if len(data) > 0: return data
        return None

    def get_pixel_size(detection):
        p1 = np.array(detection[0].polygon[0])
        p2 = np.array(detection[0].polygon[1])
        return np.linalg.norm(p2 - p1)

metrics_config = {
    "per_plant": [
        {"name": "total_root_length"},      
        {"name": "lateral_root_length"},    
        {"name": "primary_root_length"},    
        {"name": "number_of_laterals"},     
        {"name": "number_of_organs"},       
        {"name": "convex_area_hull"},       
        {"name": "root_density"},           
    ],
    "per_box": [] 
}

METRIC_SCALES = {
    "TotalRootLength": 1,
    "LateralRootLength": 1,
    "PrimaryRootLength": 1,
    "Convex_Area_Hull": 2,
    "RootDensity": -1, 
    "NumberOfLateralRoots": 0,
    "NumberOfOrgans": 0
}

def get_rsml_metadata(rsml_path):
    """
    Extrait l'objet datetime réel pour le calcul du delta,
    et le nom de l'image originale.
    """
    try:
        tree = ET.parse(rsml_path)
        root = tree.getroot()
        
        # 1. Date of acquisition (ChronoRoot format) - "captured" tag
        captured_tag = root.find(".//captured")
        dt_obj = None
        if captured_tag is not None and captured_tag.text:
            dt_obj = datetime.strptime(captured_tag.text, "%Y-%m-%dT%H:%M:%S")
            
        # 2. Original image name (if available) - "image/name" tag
        name_tag = root.find(".//image/name")
        img_name = name_tag.text if (name_tag is not None) else os.path.basename(rsml_path)
        
        return dt_obj, img_name
    except Exception as e:
        print(f"Warning XML parsing {os.path.basename(rsml_path)}: {e}")
        return None, os.path.basename(rsml_path)

def calculate_pixel_size_for_folder(plant_folder):
    box_path = os.path.dirname(plant_folder)
    metadata_path = os.path.join(box_path, 'metadata.json')
    default_px = 0.04 
    return default_px

# ==========================================
# MAIN LOOP
# Time,mainRootLength,lateralRootsLength,totalRootsLength,mainRootGrad,lateralRootsGrad,totalRootsGrad,mainRootAccel,lateralRootsAccel,totalRootsAccel,NumberOfLateralRoots,lateralRootDensity,lateralRootContDensity
# ==========================================

def process_root_system_architecture(input_pattern: str, output_csv: str):
    measures_dict = get_measures(metrics_config)
    plant_metrics = measures_dict.get("per_plant", [])
    all_data = []
    box_pixel_cache = {}

    search_path = os.path.join(input_pattern, "*", "Plant*")
    plant_folders = glob.glob(search_path)
    print(f"Found {len(plant_folders)} plant folders matching pattern: {search_path}")

    for plant_folder in plant_folders:
        path_parts = os.path.normpath(plant_folder).split(os.sep)
        
        plant_name = path_parts[-1] 
        box_name_str = path_parts[-2]   
        rpi_name = path_parts[-3]   
        
        try:
            plant_num = int(re.search(r'\d+', plant_name).group())
        except:
            plant_num = plant_name
        
        box_path = os.path.dirname(plant_folder)
        if box_path in box_pixel_cache:
            px_size = box_pixel_cache[box_path]
        else:
            px_size = calculate_pixel_size_for_folder(plant_folder)
            box_pixel_cache[box_path] = px_size

        print(f"Processing: {rpi_name} | Box {box_name_str} | {plant_name}")

        rsml_files = glob.glob(os.path.join(plant_folder, 'RSML', "TimeStep-*.rsml"))
        if not rsml_files:
             rsml_files = glob.glob(os.path.join(plant_folder, "TimeStep-*.rsml"))
        if not rsml_files:
            continue

        rsml_files.sort(key=lambda f: int(re.search(r'TimeStep-(\d+)', os.path.basename(f)).group(1)))
        
        plant_records = []
        
        start_time_ref = None

        for fpath in rsml_files:
            timestep = int(re.search(r'TimeStep-(\d+)', os.path.basename(fpath)).group(1))
            
            capture_date, original_filename = get_rsml_metadata(fpath)
            
            
            if start_time_ref is None and capture_date is not None:
                start_time_ref = capture_date 
                # ajust with timestep number start_time_ref - (timestep * 0.25 hours)
                start_time_ref = start_time_ref - pd.Timedelta(hours=timestep * 0.25)
            
            acquisition_time_float = 0.0
            time_elapsed_hours = 0.0
            
            if capture_date:
                # Acquisition Time = Heure + Minute/100 (Format ChronoRoot)
                acquisition_time_float = capture_date.hour + (capture_date.minute / 100.0)
                
                delta = capture_date - start_time_ref # time step in hours
                    
                time_elapsed_hours = delta.total_seconds() / 3600.0
            else:
                print("  [Warning] No capture date found, using timestep-based time.")
                time_elapsed_hours = timestep * 0.25 

            try:
                mtg = rsml2mtg(fpath)
                
                record = {
                    'FileName': original_filename, 
                    'TimeStep': timestep,
                    'box_name': rpi_name,
                    'img_num': box_name_str,
                    'plant_num': plant_num,
                    'Time elapsed (hours)': time_elapsed_hours,
                    'Acquisition Time': acquisition_time_float,
                    'PixelSize': px_size
                }
                
                for metric_func in plant_metrics:
                    class_name = type(metric_func).__name__ 
                    try:
                        raw_val = metric_func(mtg)
                        scale_type = METRIC_SCALES.get(class_name, 0)
                        
                        if raw_val is None:
                            record[class_name] = None
                        elif scale_type == 1:
                            record[class_name] = raw_val * px_size
                        elif scale_type == 2:
                            record[class_name] = raw_val * (px_size ** 2)
                        elif scale_type == -1:
                            record[class_name] = raw_val / px_size if px_size > 0 else 0
                        else:
                            record[class_name] = raw_val
                    except Exception:
                        record[class_name] = None
                
                plant_records.append(record)

            except Exception as e:
                print(f"  [Error] {os.path.basename(fpath)}: {e}")

        if plant_records:
            df_plant = pd.DataFrame(plant_records)
            df_plant = df_plant.sort_values('TimeStep')
            
            # Recalculer vitesses
            df_plant['dt'] = df_plant['Time elapsed (hours)'].diff()
            df_plant['dt'] = df_plant['dt'].replace(0, np.nan)
            
            df_plant['totalRootsGrad'] = df_plant['TotalRootLength'].diff() / df_plant['dt']
            df_plant['totalRootsAccel'] = df_plant['totalRootsGrad'].diff() / df_plant['dt']
            
            df_plant['lateralRootsGrad'] = df_plant['LateralRootLength'].diff() / df_plant['dt']
            df_plant['lateralRootsAccel'] = df_plant['lateralRootsGrad'].diff() / df_plant['dt']
            
            df_plant['mainRootGrad'] = df_plant['PrimaryRootLength'].diff() / df_plant['dt']
            df_plant['mainRootAccel'] = df_plant['mainRootGrad'].diff() / df_plant['dt']
            df_plant.drop(columns=['dt'], inplace=True)
            
            all_data.append(df_plant)

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        chrono_order = [
            'FileName', 'TimeStep', 
            'TotalRootLength', 'LateralRootLength', 'NumberOfLateralRoots', 
            'TotalLength', 
            'mainRootGrad', 'mainRootAccel', 'lateralRootsGrad', 'lateralRootsAccel', 'totalRootsGrad', 'totalRootsAccel',
            'NumberOfOrgans', 'Convex_Area_Hull', 'RootDensity',
            'Time elapsed (hours)', 'Acquisition Time'
        ]
        
        context_cols = ['box_name', 'img_num', 'plant_num']
        
        final_cols = []
        final_cols.extend(context_cols)
        for c in chrono_order:
            if c in final_df.columns:
                final_cols.append(c)
        remaining = [c for c in final_df.columns if c not in final_cols and c != 'PixelSize']
        final_cols.extend(remaining)
        
        final_df = final_df[final_cols]
        final_df.to_csv(output_csv, index=False)
        print(f"\nSauvegarde terminée : {output_csv}")
    else:
        print("Aucune donnée générée.")

if __name__ == "__main__":
    INPUT_ROOT = "/home/loai/Images/DataTest/ChronoRoot/Graphs/LongDay/rpi15_2020-03-12_17-01"
    OUTPUT_FILE = "global_root_analysis_exact.csv"
    process_root_system_architecture(INPUT_ROOT, OUTPUT_FILE)