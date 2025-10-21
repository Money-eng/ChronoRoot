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

import numpy as np
import cv2
from skimage.morphology import skeletonize
import os


def _as_uint8_img(arr):
    """Coerce numpy array to a cv2.imwrite-friendly uint8 image (2D or 3D)."""
    if arr is None:
        return None
    a = np.asarray(arr)
    # If boolean, scale to 0/255
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    # If float, scale/clamp to [0,255]
    elif np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, 0, 255)
        a = a.astype(np.uint8) if a.max() > 1.0 else (a * 255).astype(np.uint8)
    # If not uint8, cast
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    # Ensure 2D (H,W) or 3D (H,W,3)
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (1, 3, 4):
        return a
    # If single-channel with trailing 1, squeeze
    if a.ndim == 3 and a.shape[2] == 1:
        return a[:, :, 0]
    # Fallback: try to reduce extra dims
    return np.squeeze(a)

def plot_seg(grafo1, original, ske2):
    g, _, _, clase, _, _ = grafo1
    
    ske3 = np.zeros(list(ske2.shape)+[3], dtype='uint8')
    ske3[:,:,:] = 255
    for a in g.get_edges():
        e = g.edge(a[0],a[1])
        pos = np.where(ske2==clase[e][0])
        if clase[e][1] == 10:
            ske3[pos]=[0,180,0]
        else:
            ske3[pos]=[180,0,0]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
    ske3 = cv2.erode(ske3, kernel)
    
    c = (0,0,0)
    indices = np.where(np.all(ske3 == c, axis=-1))
    ske3[indices] = [0,180,0]
    
    return ske3


def plot_graph(grafo1, shape):
    g, pos, weight, clase, nodetype, age = grafo1
    aux = np.ones(shape, dtype='uint8')*255
    
    for pair in g.get_edges():
        v1 = pair[0]
        p1 = tuple(np.array(pos[v1], dtype='int64'))
        v2 = pair[1]
        p2 = tuple(np.array(pos[v2], dtype='int64'))
        
        edge = g.edge(v1,v2)
        if clase[edge][1] == 10:
            cv2.line(aux, p1, p2, (0, 255, 0), 2)
        else:
            cv2.line(aux, p1, p2, (255, 0, 0), 2)
            
    for i in g.get_vertices():
        p = tuple(np.array(pos[i], dtype='int64'))
        if nodetype[i] == 'Ini' or nodetype[i] == 'FTip':
            cv2.circle(aux, p, 5, (0, 200, 0), -1)
        elif nodetype[i] == 'LTip':
            cv2.circle(aux, p, 5, (0, 0, 200), -1)
        else:
            cv2.circle(aux, p, 5, (0, 200, 240), -1)
        
    return aux
        
    
def savePlotImages(name, folder, original, seg, grafo1, ske2):
    image = plot_seg(grafo1, original, ske2)
    grafo_img = plot_graph(grafo1, original.shape)

    # Normalize extension to .png to ensure a supported writer
    stem, _ = os.path.splitext(name)
    out_name = stem + ".png"

    f1 = os.path.join(folder, "img")
    os.makedirs(f1, exist_ok=True)
    path = os.path.join(f1, out_name)
    cv2.imwrite(path, _as_uint8_img(original))

    f2 = os.path.join(folder, "seg")
    os.makedirs(f2, exist_ok=True)
    path = os.path.join(f2, out_name)
    cv2.imwrite(path, _as_uint8_img(seg))

    f3 = os.path.join(folder, "labeledSeg")
    os.makedirs(f3, exist_ok=True)
    path = os.path.join(f3, out_name)
    cv2.imwrite(path, _as_uint8_img(image))

    f4 = os.path.join(folder, "graph")
    os.makedirs(f4, exist_ok=True)
    path = os.path.join(f4, out_name)
    cv2.imwrite(path, _as_uint8_img(grafo_img))

    return


def saveEmpty(name, folder, original, seg):
    image = np.ones(original.shape[:2], dtype='uint8') * 255
    grafo_img = np.ones(original.shape[:2], dtype='uint8') * 255

    # Normalize extension to .png to ensure a supported writer
    stem, _ = os.path.splitext(name)
    out_name = stem + ".png"

    f1 = os.path.join(folder, "img")
    os.makedirs(f1, exist_ok=True)
    path = os.path.join(f1, out_name)
    cv2.imwrite(path, _as_uint8_img(original))

    f2 = os.path.join(folder, "seg")
    os.makedirs(f2, exist_ok=True)
    path = os.path.join(f2, out_name)
    cv2.imwrite(path, _as_uint8_img(seg))

    f3 = os.path.join(folder, "labeledSeg")
    os.makedirs(f3, exist_ok=True)
    path = os.path.join(f3, out_name)
    cv2.imwrite(path, _as_uint8_img(image))

    f4 = os.path.join(folder, "graph")
    os.makedirs(f4, exist_ok=True)
    path = os.path.join(f4, out_name)
    cv2.imwrite(path, _as_uint8_img(grafo_img))

    return

def getCleanSeg(segFile, bbox, seed, originalSeed):
    # seg = segfile but everything else than the bounding box is zeroed
    seg = segFile.copy()
    seg[0:bbox[0],:] = 0
    seg[bbox[1]:,:] = 0
    seg[:,0:bbox[2]] = 0
    seg[:,bbox[3]:] = 0
    
    seg[0:originalSeed[1],:] = 0

    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    seg = cv2.dilate(seg, kernel)
    seg = cv2.erode(seg, kernel)
    seg = cv2.erode(seg, kernel)
    seg = cv2.dilate(seg, kernel)
    
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    j = 0
    

    if len(contour_sizes) != 0:
        ### Sorts list of contours by size from bigger to smaller
        contour_sizes.sort(key=lambda x: x[0], reverse=True)
        
        for contour in contour_sizes:
            if contour[0] < 30:
                break
            else:
                dist = cv2.pointPolygonTest(contour[1],(seed[0], seed[1]), True)
                dist = np.abs(dist)
                is_in = cv2.pointPolygonTest(contour[1],(seed[0], seed[1]), False) > 0
                
                if (dist < 30 or is_in):
                    mask = np.zeros(seg.shape, np.uint8)
                    cv2.drawContours(mask,[contour[1]], -1, 255, -1) 
                    seg2 = cv2.bitwise_and(mask, seg.copy())
                
                    return seg2, True
            
    return seg, False


def getCleanSke(seg):
    ske = np.array(skeletonize(seg // 255), dtype = 'uint8')
    
    ske = prune(ske, 5) # Removes branches with length lower than 5 pixels
    ske = trim(ske) # Removes unwanted pixels from the skeleton
    ske = prune(ske, 3) # Removes branches with length lower than 3 pixels
    ske = trim(ske) # Removes unwanted pixels from the skeleton

    bnodes, enodes = skeleton_nodes(ske) # Branch and end nodes of the skeleton 
    
    flag = False
    if len(enodes) >= 2:
        flag = True
        
    return ske, bnodes, enodes, flag


def trim(ske): ## Removes unwanted pixels from the skeleton
    
    T=[]
    T0=np.array([[-1, 1, -1], 
                 [1, 1, 1], 
                 [0, 0, 0]]) # T0 contains X0
    T2=np.array([[-1, 1, 0], 
                 [1, 1, 0], 
                 [-1, 1, 0]])
    T4=np.array([[0, 0, 0], 
                 [1, 1, 1], 
                 [-1, 1, -1]])
    T6=np.array([[0, 1, -1], 
                 [0, 1, 1], 
                 [0, 1, -1]])
    S1=np.array([[1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, -1]])
    S2=np.array([[-1, 1, -1], 
                 [1, 1, -1], 
                 [1, -1, -1]])
    S3=np.array([[-1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, 1]])
    S4=np.array([[-1, -1, -1], 
                 [-1, 1, 1], 
                 [1, 1, -1]])
    S5=np.array([[-1, 1, 1], 
                 [1, 1, -1], 
                 [-1, -1, -1]])
    S6=np.array([[1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, -1]])
    S7=np.array([[-1, -1, 1], 
                 [-1, 1, 1], 
                 [-1, 1, -1]])
    S8=np.array([[-1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, 1]])
    C1=np.array([[-1, 1, -1], 
                 [-1, 1, 1], 
                 [-1, -1, -1]])
    C2=np.array([[-1, -1, -1], 
                 [-1, 1, 1], 
                 [-1, 1, -1]])
    C3=np.array([[-1, 1, -1], 
                 [1, 1, -1], 
                 [-1, -1, -1]])
    C4=np.array([[-1, -1, -1], 
                 [1, 1, -1], 
                 [-1, 1, -1]])
    
    T.append(T0)
    T.append(T2)
    T.append(T4)
    T.append(T6)
    T.append(S1)
    T.append(S2)
    T.append(S3)
    T.append(S4)
    T.append(S5)
    T.append(S6)
    T.append(S7)
    T.append(S8)    
    T.append(C1)
    T.append(C2)
    T.append(C3)
    T.append(C4)
    
    bp = np.zeros_like(ske)
    for t in T:
        bp = cv2.morphologyEx(ske, cv2.MORPH_HITMISS, t)
        ske = cv2.subtract(ske, bp)
    
    # ske = cv2.subtract(ske, bp)
    
    return ske


def prune(skel, num_it): ## Removes branches with length lower than num_it
    orig = skel
    
    endpoint1 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [0, 1, 0]])
    
    endpoint2 = np.array([[0, 1, 0],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    endpoint4 = np.array([[0, -1, -1],
                          [1, 1, -1],
                          [0, -1, -1]])
    
    endpoint5 = np.array([[-1, -1, 0],
                          [-1, 1, 1],
                          [-1, -1, 0]])
    
    endpoint3 = np.array([[-1, -1, 1],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    endpoint6 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [1, -1, -1]])
    
    endpoint7 = np.array([[-1, -1, -1],
                          [-1, 1, -1],
                          [-1, 1, -1]])
    
    endpoint8 = np.array([[1, -1, -1],
                          [-1, 1, -1],
                          [-1, -1, -1]])
    
    
    for i in range(0, num_it):
        ep1 = skel - cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint1)
        ep2 = ep1 - cv2.morphologyEx(ep1, cv2.MORPH_HITMISS, endpoint2)
        ep3 = ep2 - cv2.morphologyEx(ep2, cv2.MORPH_HITMISS, endpoint3)
        ep4 = ep3 - cv2.morphologyEx(ep3, cv2.MORPH_HITMISS, endpoint4)
        ep5 = ep4 - cv2.morphologyEx(ep4, cv2.MORPH_HITMISS, endpoint5)
        ep6 = ep5 - cv2.morphologyEx(ep5, cv2.MORPH_HITMISS, endpoint6)
        ep7 = ep6 - cv2.morphologyEx(ep6, cv2.MORPH_HITMISS, endpoint7)
        ep8 = ep7 - cv2.morphologyEx(ep7, cv2.MORPH_HITMISS, endpoint8)
        skel = ep8
        
    end = endPoints(skel)
    kernel_size = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
    
    for i in range(0, num_it):
        end = cv2.dilate(end, kernel)
        end = cv2.bitwise_and(end, orig)
        
    return cv2.bitwise_or(end, skel)


def endPoints(skel):
    endpoint1=np.array([[1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint2=np.array([[-1, 1, -1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint3=np.array([[-1, -1, 1],
                        [-1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint4=np.array([[-1, -1, -1],
                        [1, 1, -1],
                        [-1, -1, -1]])
    
    endpoint5=np.array([[-1, -1, -1],
                        [-1, 1, 1],
                        [-1, -1, -1]])
    
    endpoint6=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [1, -1, -1]])
    
    endpoint7=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [-1, 1, -1]])
    
    endpoint8=np.array([[-1, -1, -1],
                        [-1, 1, -1],
                        [-1, -1, 1]])
    
    ep1 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint1)
    ep2 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint2)
    ep3 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint3)
    ep4 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint4)
    ep5 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint5)
    ep6 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint6)
    ep7 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint7)
    ep8 = cv2.morphologyEx(skel, cv2.MORPH_HITMISS, endpoint8)
    
    ep = ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8
    return ep


def skeleton_nodes(ske):
    """
    Extract the coordinates of branch and end nodes from a 2D skeleton image.
    Parameters
    ----------
    ske : numpy.ndarray
        2D binary skeleton image where non-zero/True values denote skeleton pixels.
    Returns
    -------
    bnodes : numpy.ndarray
        Array of shape (N, 2) with integer [x, y] coordinates (column, row) of branch points (junctions).
    enodes : numpy.ndarray
        Array of shape (M, 2) with integer [x, y] coordinates (column, row) of end points (tips).
    Notes
    -----
    - Coordinates are returned in (x, y) = (col, row) order (not (row, col)).
    - Relies on branchedPoints(ske) and endPoints(ske), which must return 2D binary masks of the same shape as 'ske'
      with ones at branch and end point locations, respectively.
    - If no points are found, the corresponding output is an empty array with shape (0, 2).
    """
    branch = branchedPoints(ske)
    end = endPoints(ske)
    
    bp = np.where(branch == 1)
    bnodes = []
    for i in range(len(bp[0])):
        bnodes.append([bp[1][i],bp[0][i]])
    
    ep = np.where(end == 1)
    enodes = []
    for i in range(len(ep[0])):
        enodes.append([ep[1][i],ep[0][i]])
    
    return np.array(bnodes), np.array(enodes)


def branchedPoints(skel):
    """
    Detect branching points in a binary skeleton using hit-or-miss morphology.
    This function scans a skeletonized image with a set of 3x3 templates
    representing typical branch configurations (cross/4-way, T-junctions, and
    Y-junctions), including their rotations. For each template, a hit-or-miss
    operation (cv2.MORPH_HITMISS) is computed and accumulated across all templates.
    Parameters
    ----------
    skel : numpy.ndarray
        2D single-channel binary skeleton image (dtype bool or uint8).
        Foreground (skeleton) pixels should be 1 or 255; background should be 0.
    Returns
    -------
    numpy.ndarray
        2D integer array of the same shape as `skel`. Each pixel contains the sum
        of matches from all templates at that location. Any non-zero value
        indicates a detected branching point. Depending on OpenCV's binary
        convention, non-zero values may be multiples of 255.
    Notes
    -----
    - Templates use values {1, 0, 2} to denote foreground, background, and
      don't-care, respectively (as required by cv2.MORPH_HITMISS).
    - For a boolean mask of branch points, threshold with `bp > 0`.
    - The input should be a thinned/skeletonized image; using a non-skeletonized
      image may result in false positives.
    - OpenCV may expect binary images with values {0, 1}. If your image uses
      {0, 255}, sums in the output may be scaled accordingly.
    Examples
    --------
    >>> bp = branchedPoints(skel)
    >>> branch_mask = bp > 0
    >>> ys, xs = np.where(branch_mask)
    """
    X=[]
    #cross X
    X0 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    X1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    X.append(X0)
    X.append(X1)
    
    #T like
    T=[]
    T0=np.array([[2, 1, 2], 
                 [1, 1, 1], 
                 [2, 2, 2]]) # T0 contains X0
    T1=np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]]) # T1 contains X1
    T2=np.array([[2, 1, 2], [1, 1, 2], [2, 1, 2]])
    T3=np.array([[1, 2, 2], [2, 1, 2], [1, 2, 1]])
    T4=np.array([[2, 2, 2], [1, 1, 1], [2, 1, 2]])
    T5=np.array([[2, 2, 1], [2, 1, 2], [1, 2, 1]])
    T6=np.array([[2, 1, 2], [2, 1, 1], [2, 1, 2]])
    T7=np.array([[1, 2, 1], [2, 1, 2], [2, 2, 1]])
    
    T.append(T0)
    T.append(T1)
    T.append(T2)
    T.append(T3)
    T.append(T4)
    T.append(T5)
    T.append(T6)
    T.append(T7)
    
    #Y like
    Y=[]
    Y0=np.array([[1, 0, 1], [0, 1, 0], [2, 1, 2]])
    Y1=np.array([[0, 1, 0], [1, 1, 2], [0, 2, 1]])
    Y2=np.array([[1, 0, 2], [0, 1, 1], [1, 0, 2]])
    Y3=np.array([[0, 2, 1], [1, 1, 2], [0, 1, 0]])
    Y4=np.array([[2, 1, 2], [0, 1, 0], [1, 0, 1]])
    Y5 = np.rot90(Y3)
    Y6 = np.rot90(Y4)
    Y7 = np.rot90(Y5)
    
    Y.append(Y0)
    Y.append(Y1)
    Y.append(Y2)
    Y.append(Y3)
    Y.append(Y4)
    Y.append(Y5)
    Y.append(Y6)
    Y.append(Y7)
    
    bp = np.zeros(skel.shape, dtype=int)
    for x in X:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, x)
    for y in Y:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, y)
    for t in T:
        bp = bp + cv2.morphologyEx(skel, cv2.MORPH_HITMISS, t)
        
    return bp