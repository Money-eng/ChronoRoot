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

from graph.ChronoRoot import ChronoRootAnalyzer
import os
from rsml import rsml2mtg, mtg2rsml
from pathlib import Path
from openalea.mtg import MTG
import sys

# Ajoute la racine du projet: <...>/RSA_reconstruction
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    conf1 = {}
    file = exec(open('/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/config.conf').read(), conf1)
    conf2 = {}
    file = exec(open('/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/cnns.conf').read(), conf2)

    conf = {**conf1, **conf2}

    #if not args.imgpath:
      #  pass
    #else:
     #   conf['Path'] = args.imgpath

    #if not args.segpath:
     #   pass
    #else:
     #   conf['SegPath'] = args.segpath


    # List all folder in the input directory
    input_dir = Path("/home/loai/Documents/code/RSMLExtraction/temp/input")
    box_dir = list(input_dir.glob("*/*"))
    box_dir = [p for p in box_dir if p.is_dir()]
    box_dir.sort()
    box_parent_dir = [p.parent for p in box_dir]
    
    # In each folder, image is 22_registered_stack.tif
    # Segmentation is 40_date_map.tif
    # RSML is 61_graph.rsml
    # Load them and pass to ChronoRootAnalyzer
    for box in box_dir:
        conf['Project'] = "/home/loai/Documents/code/RSMLExtraction/temp/output/" + box.parent.name + "/" + box.name
        os.makedirs(conf['Project'], exist_ok=True)
        images_path = box / "22_registered_stack.tif"
        seg_path = box / "40_date_map.tif"
        rsml_path = box / "61_graph.rsml"
        gt_rsml_path = rsml2mtg(rsml_path)
        time_list = [float(t) for t in gt_rsml_path.graph_properties().get('metadata',{}).get("observation-hours", {}).split(",")]
        if images_path.exists() and seg_path.exists() and rsml_path.exists():
            print(f"Processing folder: {box}")
            from tifffile import TiffFile
            import numpy as np
            # load images and segmentations as lists of numpy arrays
            images = []
            segFiles = []
            with TiffFile(images_path) as tif:
                for page in tif.pages:
                    images.append(page.asarray())
            with TiffFile(seg_path) as tif:
                for page in tif.pages:
                    max_value = page.asarray().max()
                    max_value = int(max_value)
                    for i in range(max_value):  # Iterate over the three channels
                        seg_slice = page.asarray()
                        seg = (seg_slice <= i + 1) & (seg_slice > 0)
                        seg = seg.astype(np.uint8) * 255
                        segFiles.append(seg.copy())
            conf['captureTimes'] = time_list
            ChronoRootAnalyzer(conf = conf, images = images, segFiles = segFiles, rsml_path = rsml_path)
            
            from glob import glob
            import re
            # in conf['Project'] regroup all rsml files that have the same plant value (P1, P2, ..., P5)
            list_rsml_files = glob(str(conf['Project']) + "/*.rsml")
            patterns = ['P1', 'P2', 'P3', 'P4', 'P5']
            matched_files = {}
            for pattern in patterns:
                matched_files[pattern] = [f for f in list_rsml_files if pattern in os.path.basename(f)]
            print(matched_files)
            
            def extract_time_index(filepath: str) -> int:
                name = os.path.basename(filepath)
                m = re.search(r'[_-][Ii](\d+)', name)
                if m:
                    try:
                        return int(m.group(1))
                    except ValueError:
                        pass

                nums = re.findall(r'(\d+)', name)
                if nums:
                    return int(nums[-1])
                return -1

            grouped_by_plant_and_time = {}
            for plant, files in matched_files.items():
                
                files_sorted = sorted(files, key=lambda f: (extract_time_index(f), f))
                grouped_by_plant_and_time[plant] = files_sorted

            # At each time point, merge the rsml files of all plants into a single rsml file using rsml library
            for time_point in range(len(grouped_by_plant_and_time['P1'])):
                merged_rsml = None
                for plant in patterns:
                    try:
                        rsml_file = grouped_by_plant_and_time[plant][time_point]
                    except IndexError:
                        continue
                    # crop the rsml from <plant>_ to </plant> and add it at the end of merged_rsml
                    if merged_rsml is None:
                        merged_rsml = open(rsml_file, 'r').read()
                        # # get last id= number in the file
                        # ids = re.findall(r'id="(\d+)"', merged_rsml)
                        # if ids:
                        #     max_id = max([int(i) for i in ids])
                        #     next_id = max_id + 1
                    else:
                        xml_content = open(rsml_file, 'r').read()
                        # find the content between <plant> and </plant>
                        plant_content = re.search(r'<plant.*?>.*?</plant>', xml_content, re.DOTALL)
                        if plant_content:
                            merged_rsml = merged_rsml.replace('</rsml>', plant_content.group(0) + '\n</rsml>')
                            # # increment all id= numbers in plant_content by next_id
                            # for match in re.finditer(r'id="(\d+)"', plant_content.group(0)):
                            #     old_id = int(match.group(1))
                            #     new_id = old_id + next_id
                            #     merged_rsml = merged_rsml.replace(f'id="{old_id}"', f'id="{new_id}"')
                            # # update next_id
                            # ids = re.findall(r'id="(\d+)"', merged_rsml)
                            # if ids:
                            #     max_id = max([int(i) for i in ids])
                            #     next_id = max_id + 1
                # save merged_rsml to conf['Project'] + /merged_time_<time_point>.rsml
                with open(conf['Project'] + f"/merged_time_{time_point}.rsml", 'w') as f:
                    f.write(merged_rsml)
                mtg = conf['Project'] + f"/merged_time_{time_point}.rsml"
                
                mtg = rsml2mtg(mtg)
                print(f"Merged rsml saved to: {mtg}")
                # remove old file
                os.remove(conf['Project'] + f"/merged_time_{time_point}.rsml")
                # save it again to make sure it is valid
                mtg2rsml(mtg, conf['Project'] + f"/merged_time_{time_point}.rsml")

        # delete everything that is not a file like 'merged_time_*.rsml'
        for f in os.listdir(conf['Project']):
            if os.path.isfile(os.path.join(conf['Project'], f)):
                if not re.match(r'merged_time_\d+\.rsml', f):
                    os.remove(os.path.join(conf['Project'], f))
            else:
                # remove directory and all its content
                import shutil
                shutil.rmtree(os.path.join(conf['Project'], f))       