import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from apls import APLSMetric

# ==========================================
# 1. Remplacement de sknw (Version stable)
# ==========================================
def skeleton_to_graph(skeleton):
    """
    Convertit un squelette binaire en graphe NetworkX.
    Version corrigée pour gérer les changements dynamiques de topologie (IndexError).
    """
    pixels = np.column_stack(np.where(skeleton))
    G = nx.Graph()
    
    for r, c in pixels:
        G.add_node((r, c))
        
    for r, c in pixels:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in G.nodes:
                    dist = np.sqrt(dr**2 + dc**2)
                    # On ajoute l'arête (NetworkX ignore les doublons dans Graph simple)
                    G.add_edge((r, c), (nr, nc), length=dist)
    
    # 3. Simplification (retrait des nœuds de passage)
    # On identifie les candidats initiaux
    nodes_to_check = [n for n in G.nodes if G.degree(n) == 2]
    
    for n in nodes_to_check:
        if n not in G.nodes:
            continue
            
        if G.degree(n) != 2:
            continue
            
        neighbors = list(G.neighbors(n))
        
        if len(neighbors) != 2:
            continue
            
        u, v = neighbors[0], neighbors[1]
        
        if u == v:
            continue
            
        w1 = G[u][n]['length']
        w2 = G[n][v]['length']
        
        if G.has_edge(u, v):
            G[u][v]['length'] += w1 + w2
        else:
            G.add_edge(u, v, length=w1 + w2)
            
        G.remove_node(n)
        
    for node in G.nodes:
        r, c = node
        G.nodes[node]['y'] = r
        G.nodes[node]['x'] = c
            
    return G

def mask_to_graph(binary_mask):
    # 1. Squelettisation
    skeleton = skeletonize(binary_mask.astype(bool))
    
    # 2. Conversion avec notre nouvelle fonction (plus de sknw)
    G = skeleton_to_graph(skeleton)
    
    return G

# ==========================================
# NOUVELLE FONCTION : Augmentation du graphe
# ==========================================
def inject_midpoints(G, interval_pixels):
    """
    Injecte des nœuds intermédiaires le long des arêtes trop longues.
    C'est CRUCIAL pour l'APLS sur des segmentations discontinues.
    """
    G_aug = G.copy()
    edges_to_remove = []
    edges_to_add = []
    
    # On itère sur toutes les arêtes existantes
    for u, v, data in G.edges(data=True):
        length = data.get('length', 0)
        
        # Si l'arête est plus longue que l'intervalle, on la découpe
        if length > interval_pixels:
            edges_to_remove.append((u, v))
            
            # Récupérer positions
            p_u = np.array([G.nodes[u]['y'], G.nodes[u]['x']])
            p_v = np.array([G.nodes[v]['y'], G.nodes[v]['x']])
            
            # Combien de segments ?
            num_segments = int(np.ceil(length / interval_pixels))
            
            # Création des points intermédiaires
            prev_node = u
            for i in range(1, num_segments):
                # Interpolation linéaire (t varie de 0 à 1)
                t = i / num_segments
                new_pos = p_u + t * (p_v - p_u)
                
                # Créer un ID unique pour le nouveau nœud (tuple float)
                new_node_id = (new_pos[0], new_pos[1])
                
                # Ajouter le nœud avec ses attributs
                G_aug.add_node(new_node_id, y=new_pos[0], x=new_pos[1])
                
                # Calculer la distance du petit segment
                seg_len = np.linalg.norm(new_pos - np.array([G_aug.nodes[prev_node]['y'], G_aug.nodes[prev_node]['x']]))
                
                # Ajouter l'arête
                edges_to_add.append((prev_node, new_node_id, seg_len))
                prev_node = new_node_id
            
            # Connecter le dernier point intermédiaire au nœud final v
            last_seg_len = np.linalg.norm(p_v - np.array([G_aug.nodes[prev_node]['y'], G_aug.nodes[prev_node]['x']]))
            edges_to_add.append((prev_node, v, last_seg_len))
            
    # Appliquer les modifications
    G_aug.remove_edges_from(edges_to_remove)
    for u, v, l in edges_to_add:
        G_aug.add_edge(u, v, length=l)
        
    return G_aug

