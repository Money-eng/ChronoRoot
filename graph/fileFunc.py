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
import pathlib
import re
import cv2
import numpy as np

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def loadPath(search_path, ext = '*.*'):
    data_root = pathlib.Path(search_path)
    all_files = list(data_root.glob(ext))
    all_files = [str(path) for path in all_files]
    all_files.sort(key = natural_key)
    print('Number of files found:', len(all_files))
    return all_files


def createResultFolder(conf):
    """
    Create a result directory tree for a run under conf['Project'].
    This function attempts to:
    - Ensure the project directory (conf['Project']) exists.
    - Create a unique results subdirectory named "Results <i>" with the smallest
        available i, starting at 0. If creation fails for 0..20, it raises.
    - Inside the chosen results directory, create:
        - "Graphs"
        - "RSML"
        - Optionally (when conf['SaveImages'] is True): "Imagenes" with
            subfolders "img", "seg", "labeledSeg", and "graph".
    Most directory-creation errors are silently ignored (e.g., when directories
    already exist), except for the capped attempts to create "Results <i>".
    Args:
            conf (dict): Configuration mapping with required keys:
                    - 'Project' (str): Path to the project directory where results will be created.
                    - 'SaveImages' (bool): Whether to create the "Imagenes" folder and its subfolders.
    Returns:
            tuple[str, str, str, str]: Paths to:
                    - saveFolder: The created "Results <i>" directory.
                    - graphsPath: The "Graphs" subdirectory.
                    - imagePath: The "Imagenes" subdirectory (if created).
                    - rsmlPath: The "RSML" subdirectory.
    Raises:
            Exception: If a "Results <i>" directory cannot be created by the time i reaches 20.
            KeyError: If required keys ('Project', 'SaveImages') are missing from conf.
            UnboundLocalError: If conf['SaveImages'] is False (imagePath is not defined but still returned).
    Notes:
    - Although the search loop runs up to i == 99, failure is reported early at i == 20.
    - Broad exception handling means that permission errors or other OS errors during
        intermediate directory creation (other than the capped "Results <i>" attempts) are suppressed.
    """
    try:
        os.mkdir(conf['Project'])
    except:
        pass

    for i in range(0,100):
        saveFolder = os.path.join(conf['Project'], "Results %s" %(i))
        try:
            os.mkdir(saveFolder)
            break
        except:
            if i == 20:
                raise Exception('Could not create Results Folder')
            pass
    
    graphsPath =  os.path.join(saveFolder, "Graphs")
    try:
        os.mkdir(graphsPath)
    except:
        pass
    
    if conf['SaveImages']:
        imagePath = os.path.join(saveFolder, "Imagenes")
        try:
            os.mkdir(imagePath)
            
            f1 = os.path.join(imagePath, "img")
            f2 = os.path.join(imagePath, "seg")
            f3 = os.path.join(imagePath, "labeledSeg")
            f4 = os.path.join(imagePath, "graph")
            
            os.mkdir(f1)
            os.mkdir(f2)
            os.mkdir(f3)
            os.mkdir(f4)
        except:
            pass
        
    rsmlPath = os.path.join(saveFolder, "RSML")
    try:
        os.mkdir(rsmlPath)
    except:
        pass
    
    return saveFolder, graphsPath, imagePath, rsmlPath


def selectROI(image):
    y, x = image.shape[0:2]
    y = y//4
    x = x//4
    
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", x, y)
    
    r = cv2.selectROI("Image",image)

    cv2.waitKey()
    cv2.destroyAllWindows()
    return r

pos = []


def mouse_callback(event,x,y,flags,param):
    global pos
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = [(x, y)]
        print(pos)
        
def selectSeed(images):
    n = len(images)
    i = 0
    image = images[i]
    
    global pos
    y, x = image.shape[0:2]
    y = y//2
    x = x//2
    
    clone = image.copy()
    
    cv2.namedWindow("Image",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", x, y)
    cv2.setMouseCallback('Image',mouse_callback) #Mouse callback
    
    while True:
        if pos != []:
            cv2.circle(clone,pos[0],6,[255,0,0],-1)
        
    	# display the image and wait for a keypress
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF
     
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("+"):
            i = (i+1)%n
            image = images[i]
            clone = image.copy()
        
        if key == ord("r"):
            clone = image.copy()
     
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        
        elif key == 13:
            break
    
    cv2.destroyAllWindows()
    return pos


def getROIandSeed(conf, images, segFiles):
    """
    Interactively select a region of interest (ROI) and a seed from a time series of images.
    This function displays the last available image (synchronized by the shorter of `images`
    and `segFiles`) to let the user select an ROI. It then crops that ROI from a small set of
    subsampled frames and asks the user to select a seed from those crops. The ROI bounds and
    the selected seed are returned.
    Parameters
    ----------
    conf : dict
        Configuration dictionary. Must contain:
        - 'timeStep' (int | float): Temporal resolution in minutes between consecutive frames.
    images : Sequence[str]
        Ordered file paths to the original images (chronological order expected).
    segFiles : Sequence[str]
        File paths to corresponding segmentation results. Only the length is used to compute
        the number of synchronized frames.
    Returns
    -------
    p : numpy.ndarray
        Array of shape (4,) with integer pixel indices [row_start, row_end, col_start, col_end]
        delimiting the selected ROI in the original image coordinate system.
    seed : Any
        The seed selection as returned by `selectSeed(crops)`. Typically represents a point
        or set of points within the ROI; the exact structure depends on the implementation
        of `selectSeed`.
    Notes
    -----
    - The ROI is selected on the image at index min(len(images), len(segFiles)) - 1.
    - The number of days is estimated as dia = (24 * 60) / conf['timeStep'].
      The number of samples `c` is computed as max(1, floor(N / dia)), where N is the
      synchronized frame count.
    - Cropped frames presented for seed selection are taken at indices [0, 100, 200, ...]
      up to `c - 1`. These indices must exist in `images`.
    - This function requires user interaction via `selectROI` and `selectSeed`, and relies
      on OpenCV (`cv2`) for image I/O.
    Raises
    ------
    KeyError
        If 'timeStep' is missing from `conf`.
    IndexError
        If `images` or `segFiles` are empty, or if the subsampled indices exceed the range
        of `images`.
    RuntimeError | ValueError
        If image loading fails or interactive selection functions (`selectROI`/`selectSeed`)
        cannot complete successfully.
    """
    N1 = len(images) # get number of images
    N2 = len(segFiles) # get number of segmentations

    N = min(N1,N2) # use the shortest sequence length
    
    original = cv2.imread(images[N-1]) # load the last available image
    
    r = selectROI(original) # let user select ROI
    print('Selected ROI:', r)
    p = np.array([int(r[1]),int(r[1]+r[3]), int(r[0]),int(r[0]+r[2])]) # define ROI bounds - y_min, y_max, x_min, x_max
    
    crops = []
    
    t = conf['timeStep'] #always in minutes
    dia = 24*60/t # number of frames per day
    c = int(N//dia) # number of samples to take
    c = max(1, c) # at least one sample
    
    for i in range(0, c): # get c samples
        P2 = int(i*100) # every 100 frames
        img = cv2.imread(images[P2]) # load image
        boundingBox = img[p[0]:p[1],p[2]:p[3]] # crop it
        crops.append(boundingBox) # add to list
    
    print('Number of crops for seed selection:', len(crops))
    seed = selectSeed(crops) # let user select seed from crops
    
    return p, seed