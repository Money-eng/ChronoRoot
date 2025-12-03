import networkx as nx
import numpy as np
from scipy.spatial import KDTree

# Généré
class APLSMetric:
    def __init__(self, G_ground_truth, G_proposal, snap_buffer_meters=4.0):
        """
        Initialise le calculateur APLS.
        
        Args:
            G_ground_truth (nx.Graph): Graphe de vérité terrain (doit avoir des attributs 'pos' ou géométrie).
            G_proposal (nx.Graph): Graphe prédit par le modèle.
            snap_buffer_meters (float): Distance max pour associer un nœud (4m par défaut selon le papier).
        """
        self.G_gt = G_ground_truth
        self.G_prop = G_proposal
        self.snap_buffer = snap_buffer_meters

    def _get_node_positions(self, G):
        """Extrait les positions (x, y) des nœuds pour le snapping."""
        nodes = list(G.nodes())
        # On suppose que les nœuds ont un attribut 'pos'=(x,y) ou sont des tuples (x,y)
        # Dans SpaceNet/OSMnx, les nœuds ont souvent des attributs 'x' et 'y'
        coords = []
        for n in nodes:
            if 'x' in G.nodes[n] and 'y' in G.nodes[n]:
                coords.append((G.nodes[n]['x'], G.nodes[n]['y']))
            else:
                # Fallback si le nœud est lui-même la coordonnée
                coords.append(n) 
        return nodes, np.array(coords)

    def _calculate_single_direction_apls(self, G_source, G_target):
        """
        Calcule le score APLS dans une direction (ex: GT -> Proposal).
        """
        source_nodes, source_coords = self._get_node_positions(G_source)
        target_nodes, target_coords = self._get_node_positions(G_target)

        if len(target_nodes) == 0:
            return 0.0 # Score nul si le graphe cible est vide

        # Création d'un KDTree pour trouver rapidement les nœuds les plus proches
        tree = KDTree(target_coords)

        paths_count = 0
        sum_differences = 0.0

        # On itère sur un sous-ensemble de paires pour éviter une explosion combinatoire
        # Le papier suggère d'utiliser des "Control Nodes" (intersections + midpoints)
        # Ici, nous itérons sur tous les nœuds pour simplifier, mais sur de gros graphes, il faut échantillonner.
        nodes_of_interest = source_nodes 

        for i, u_src in enumerate(nodes_of_interest):
            # Trouver le nœud correspondant le plus proche dans le graphe cible (Snapping)
            dist_u, idx_u = tree.query(source_coords[i])
            u_tgt = target_nodes[idx_u] if dist_u <= self.snap_buffer else None

            for j, v_src in enumerate(nodes_of_interest):
                if i >= j: continue # On évite les doublons et l'identité
                
                # Snapping du deuxième nœud
                dist_v, idx_v = tree.query(source_coords[j])
                v_tgt = target_nodes[idx_v] if dist_v <= self.snap_buffer else None

                # Calcul du chemin dans la source (Vérité terrain)
                try:
                    len_src = nx.shortest_path_length(G_source, u_src, v_src, weight='length') # Djikstra par défaut
                except nx.NetworkXNoPath:
                    continue # S'il n'y a pas de chemin dans la source, on ignore cette paire

                paths_count += 1

                # Si le snapping a échoué (hors buffer), pénalité maximale
                if u_tgt is None or v_tgt is None:
                    sum_differences += 1.0
                    continue

                # Calcul du chemin dans la cible (Proposition)
                try:
                    len_tgt = nx.shortest_path_length(G_target, u_tgt, v_tgt, weight='length')
                    
                    # Formule APLS (Equation 3 du papier)
                    diff = abs(len_src - len_tgt) / len_src
                    sum_differences += min(1.0, diff)
                    
                except nx.NetworkXNoPath:
                    # Pénalité maximale si le chemin n'existe pas dans la proposition
                    sum_differences += 1.0

        if paths_count == 0:
            return 0.0
            
        return 1 - (sum_differences / paths_count)

    def compute(self):
        """
        Calcule le score APLS détaillé.
        """
        # Part 1: RECALL (Est-ce que le GT est dans la Pred ?)
        # C'est CELUI-CI qui t'intéresse pour ton cas.
        recall_score = self._calculate_single_direction_apls(self.G_gt, self.G_prop)
        
        # Part 2: PRECISION (Est-ce que la Pred est réelle ?)
        precision_score = self._calculate_single_direction_apls(self.G_prop, self.G_gt)

        # Moyenne Harmonique (F1 Global)
        if recall_score + precision_score == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * (recall_score * precision_score) / (recall_score + precision_score)
            
        return {
            'recall_gt_to_pred': recall_score,  # <--- Regarde ce chiffre
            'precision_pred_to_gt': precision_score,
            'apls_f1': f1_score
        }