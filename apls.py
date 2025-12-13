import networkx as nx
import numpy as np
from scipy.spatial import KDTree

class APLSMetric:
    def __init__(self, G_ground_truth, G_prediction, snap_buffer_meters=4.0):
        self.G_gt = G_ground_truth
        self.G_pred = G_prediction
        self.snap_buffer = snap_buffer_meters 

    def _get_node_positions(self, G):
        nodes = list(G.nodes())
        coords = []
        for n in nodes:
            if 'x' in G.nodes[n] and 'y' in G.nodes[n]:
                coords.append((G.nodes[n]['x'], G.nodes[n]['y']))
            else:
                coords.append(n) 
        return nodes, np.array(coords)

    def _calculate_single_direction_apls(self, G_source, G_target):
        source_nodes, source_coords = self._get_node_positions(G_source)
        target_nodes, target_coords = self._get_node_positions(G_target)

        if len(target_nodes) == 0: return 0.0

        tree = KDTree(target_coords) # Create KDTree for fast nearest neighbor search
        paths_count = 0
        sum_differences = 0.0
        nodes_of_interest = source_nodes 

        import tqdm 
        for i, u_src in enumerate(tqdm.tqdm(nodes_of_interest)):
            dist_u, idx_u = tree.query(source_coords[i])  # Query nearest neighbor in target
            u_tgt = target_nodes[idx_u] if dist_u <= self.snap_buffer else None # if within snap buffer else None

            for j, v_src in enumerate(nodes_of_interest):
                if i >= j: continue 
                
                dist_v, idx_v = tree.query(source_coords[j]) # Query nearest neighbor in target
                v_tgt = target_nodes[idx_v] if dist_v <= self.snap_buffer else None # if within snap buffer else None

                try:
                    len_src = nx.shortest_path_length(G_source, u_src, v_src, weight='length') # get length in source graph between u_src and v_src
                except nx.NetworkXNoPath:
                    continue 

                paths_count += 1

                if u_tgt is None or v_tgt is None:
                    sum_differences += 1.0 # No match found within snap buffer
                    continue

                try:
                    len_tgt = nx.shortest_path_length(G_target, u_tgt, v_tgt, weight='length') # get length in target graph between u_tgt and v_tgt
                    diff = abs(len_src - len_tgt) / len_src # relative difference
                    sum_differences += min(1.0, diff) # Cap at 1.0
                except nx.NetworkXNoPath:
                    sum_differences += 1.0 # Max penalty: path cut

        if paths_count == 0: return 0.0 # No paths to compare
        return 1 - (sum_differences / paths_count) # APLS score
    
    def compute(self):
        recall = self._calculate_single_direction_apls(self.G_gt, self.G_pred)
        precision = self._calculate_single_direction_apls(self.G_pred, self.G_gt)
        if recall + precision == 0: f1 = 0.0
        else: f1 = 2 * (recall * precision) / (recall + precision)
        return {'recall': recall, 'precision': precision, 'f1': f1}
        
        
# --- HELPER FUNCTIONS ---

def create_simple_line(length=10.0, coords=[(0,0), (10,0)]):
    """Creates a simple 2-node graph."""
    G = nx.Graph()
    G.add_node(coords[0], x=coords[0][0], y=coords[0][1])
    G.add_node(coords[1], x=coords[1][0], y=coords[1][1])
    G.add_edge(coords[0], coords[1], length=float(length))
    return G

def create_grid(rows=3, cols=3, spacing=10.0, offset_x=0.0, offset_y=0.0):
    """
    Creates a Manhattan grid graph.
    Nodes are positioned based on `spacing` and `offset`.
    Weights are set to Euclidean distance.
    """
    G = nx.grid_2d_graph(rows, cols)
    
    # Map (i, j) tuple to spatial (x, y) coordinates
    mapping = {}
    for n in G.nodes():
        x = n[0] * spacing + offset_x
        y = n[1] * spacing + offset_y
        mapping[n] = (x, y) # Rename node to coord tuple
    
    G = nx.relabel_nodes(G, mapping)

    # Set attributes
    for u, v in G.edges():
        G.edges[u, v]['length'] = spacing # Orthogonal grid, length = spacing
    
    for n in G.nodes():
        G.nodes[n]['x'] = n[0]
        G.nodes[n]['y'] = n[1]
        
    return G

# --- TEST SUITE ---
import unittest
class TestAPLS_Standardized(unittest.TestCase):
    def setUp(self):
        self.buffer = 4.0 
        
    def test_01_identity(self):
        """Test: Perfect match on a simple line."""
        G = create_simple_line(length=50.0)
        res = APLSMetric(G, G).compute()
        self.assertEqual(res['f1'], 1.0, "Identity should result in F1=1.0")

    def test_02_buffer_limit_success(self):
        """Test: Prediction shifted by 3m (Inside 4m buffer) -> Success."""
        G_gt = create_simple_line()
        G_pred = create_simple_line(coords=[(3,0), (13,0)]) # Shifted +3 X
        
        res = APLSMetric(G_gt, G_pred, snap_buffer_meters=4.0).compute()
        self.assertEqual(res['f1'], 1.0, "Shift within buffer should be tolerated")

    def test_03_buffer_limit_fail(self):
        """Test: Prediction shifted by 5m (Outside 4m buffer) -> Fail."""
        G_gt = create_simple_line()
        G_pred = create_simple_line(coords=[(5,0), (15,0)]) # Shifted +5 X
        
        res = APLSMetric(G_gt, G_pred, snap_buffer_meters=4.0).compute()
        self.assertEqual(res['f1'], 0.0, "Shift outside buffer should result in 0 score")

    def test_04_length_penalty(self):
        """Test: Geometry matches, but length is wrong (Shortcut)."""
        G_gt = create_simple_line(length=100.0)
        G_pred = create_simple_line(length=10.0) # 10x shorter
        
        res = APLSMetric(G_gt, G_pred).compute()
        # GT->Pred diff: |100-10|/100 = 0.9 penalty -> Score 0.1
        self.assertAlmostEqual(res['recall'], 0.1, places=2)

    def test_05_grid_perfect(self):
        """
        Test: 3x3 Grid Identity.
        Validates that the metric handles multiple paths/loops correctly.
        """
        G = create_grid(rows=3, cols=3, spacing=10.0)
        res = APLSMetric(G, G).compute()
        self.assertEqual(res['f1'], 1.0, "Grid identity should be 1.0")

    def test_06_grid_shifted_global(self):
        """
        Test: Entire 3x3 Grid shifted by 2m (within 4m buffer).
        Verifies that snapping works for ALL nodes in the network.
        """
        G_gt = create_grid(rows=3, cols=3, spacing=10.0, offset_x=0.0)
        G_pred = create_grid(rows=3, cols=3, spacing=10.0, offset_x=2.0)
        
        res = APLSMetric(G_gt, G_pred, snap_buffer_meters=4.0).compute()
        self.assertEqual(res['f1'], 1.0, "Shifted grid (within buffer) should score 1.0")

    def test_07_grid_missing_horizontal_edges(self):
        """
        Test: Topology Break.
        GT: Full 3x3 grid.
        Pred: Same grid but missing ALL horizontal edges (disconnected columns).
        """
        G_gt = create_grid(rows=3, cols=3)
        G_pred = create_grid(rows=3, cols=3)
        
        # Remove horizontal edges in Pred: edges where y is same, x differs
        edges_to_remove = []
        for u, v in G_pred.edges():
            if u[1] == v[1]: # Same Y coordinate = horizontal edge
                edges_to_remove.append((u, v))
        G_pred.remove_edges_from(edges_to_remove)
        
        res = APLSMetric(G_gt, G_pred).compute()
        
        # RECALL (GT -> Pred): 
        # Many paths existing in GT (horizontal) are impossible in Pred.
        # Should be very low.
        print(f"\n[Test 07 Debug] Grid Broken Recall: {res['recall']}")
        self.assertLess(res['recall'], 0.5, "Recall should drop heavily on disconnected grid")

    def test_08_grid_different_densities(self):
        """
        Test: Dense Grid (GT) vs Sparse Grid (Pred).
        GT: 5x5 nodes. Pred: 3x3 nodes (covering same area roughly, but fewer nodes).
        This simulates a model missing intermediate intersections.
        """
        # GT is dense: spacing 10 over 40x40 area
        G_gt = create_grid(rows=5, cols=5, spacing=10.0) 
        
        # Pred is sparse: spacing 20 over 40x40 area (only matches intersection nodes)
        G_pred = create_grid(rows=3, cols=3, spacing=20.0)
        
        res = APLSMetric(G_gt, G_pred, snap_buffer_meters=4.0).compute()
        
        # Note: The corners match (0,0), (40,40), but (10,10) in GT has no match in Pred (spacing 20).
        # So Snapping will fail for intermediate nodes.
        self.assertLess(res['f1'], 0.6, "Mismatched node density should lower score due to snapping failures")

    def test_09_line_sep(self):
        """
        Test: Dense Grid (GT) vs Dense Grid (Pred).
        GT: 6x6 nodes. Pred: 6x6 nodes but separated into two halves.
        This simulates a model missing intermediate intersections.
        """
        G_gt = create_grid(rows=6, cols=6, spacing=10.0) 
        G_pred = create_grid(rows=6, cols=6, spacing=10.0)
        
        # disconnect into two halves
        edges_to_remove = []
        for u, v in G_pred.edges():
            if (u[0] < 30 and v[0] >= 30) or (u[0] >= 30 and v[0] < 30):
                edges_to_remove.append((u, v))
        G_pred.remove_edges_from(edges_to_remove)   
        res = APLSMetric(G_gt, G_pred, snap_buffer_meters=4.0).compute()
        
        # total number of tuples in gt is (6*6)*(6*6-1)/2 = 630
        # the separated graph can only match paths within each half
        # each half has (3*6)*(3*6-1)/2 = 153
        # so total matched paths is 2 * 153 = 306
        # recall should be 306/630 = 0.4857
        # precision should be 1.0 since all predicted paths are valid
        # and f1 = 2 * (precision * recall) / (precision + recall) = 2 * (1.0 * recall) / (1.0 + recall)
        expected_recall = 306.0 / 630.0
        expected_precision = 1.0
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        self.assertAlmostEqual(res['recall'], expected_recall, places=4, msg="Recall should match expected value for separated graph")
        self.assertAlmostEqual(res['precision'], expected_precision, places=4, msg="Precision should be 1.0 for separated graph")
        self.assertAlmostEqual(res['f1'], expected_f1, places=4, msg="F1 should match expected value for separated graph")
        

if __name__ == '__main__':
    unittest.main()
    
    # from skimage.morphology import skeletonize
    # binary_img = cv2.imread('/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/ChronoRoot/logs/model_SegNet/val_gt_epoch_170.png', cv2.IMREAD_GRAYSCALE)
    
    # import scipy.ndimage as ndi
    # labeled_img, num_features = ndi.label(binary_img > 0)
    # for i in range(1, num_features + 1):
    #     skeleton = skeletonize(labeled_img == i)

    #     # keep only one connected component (largest)

    #     G = skeleton_to_graph_sampled(skeleton, sample_dist=5)
    #     G_sample = skeleton_to_graph_sampled(skeleton, sample_dist=6.0) # to test different sampling

    #     # compute APLS between G and G_sample
    #     metric = APLSMetric(G, G_sample, snap_buffer_meters=4.0)
    #     score = metric.compute()
    #     print(f"APLS Score between dense and sampled graph: {score}")
