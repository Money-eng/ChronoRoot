import numpy as np
import networkx as nx


def skeleton_to_graph_sampled(skeleton, sample_dist=10.0):
    pixels = np.column_stack(np.where(skeleton))
    G = nx.Graph()

    for r, c in pixels:
        G.add_node((r, c), y=r, x=c)

    # Connexions (8-voisinage)
    for r, c in pixels:
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in G.nodes:
                    # On évite les doublons d'arêtes
                    if not G.has_edge((r, c), (nr, nc)):
                        dist = np.sqrt(dr ** 2 + dc ** 2)
                        G.add_edge((r, c), (nr, nc), length=dist)

    G_simplified = G.copy()

    key_nodes = [n for n in G.nodes if G.degree(n) != 2]

    # if there are no 2-degree nodes, pick an arbitrary node to start
    if not key_nodes and len(G.nodes) > 0:
        key_nodes = [list(G.nodes)[0]]

    visited_edges = set()

    for start_node in key_nodes:
        # for each neighbor of the start_node
        for neighbor in list(G.neighbors(start_node)):

            if tuple(sorted((start_node, neighbor))) in visited_edges:
                continue

            path = [start_node,
                    neighbor]  # 'path' will contain the entire line pixel by pixel: [IntersectionA, p1, p2, ..., pN, IntersectionB]
            visited_edges.add(tuple(sorted((start_node, neighbor))))

            curr = neighbor
            prev = start_node

            # While we are on a "line" (degree 2), we move forward
            while G.degree(curr) == 2:
                nbrs = list(G.neighbors(curr))
                # Find the next node (the one that is not where we came from)
                next_node = nbrs[0] if nbrs[0] != prev else nbrs[1]

                if next_node == start_node:  # Closed loop on itself
                    path.append(next_node)
                    break

                path.append(next_node)
                visited_edges.add(tuple(sorted((curr, next_node))))

                prev = curr
                curr = next_node

                if curr in key_nodes:  # We have reached another intersection
                    break

            if len(path) <= 3:
                continue

            cumul_dist = 0
            last_kept_idx = 0

            for i in range(len(path) - 1):
                if G_simplified.has_edge(path[i], path[i + 1]):
                    G_simplified.remove_edge(path[i], path[i + 1])

            for i in range(1, len(path) - 1):
                G_simplified.remove_node(path[i])

            # Reconnect by skipping nodes
            for i in range(1, len(path)):
                u, v = path[i - 1], path[i]

                step_dist = G[u][v]['length']
                cumul_dist += step_dist

                if cumul_dist >= sample_dist and i < len(path) - 1:
                    node_to_keep = path[i]
                    prev_node_kept = path[last_kept_idx]

                    G_simplified.add_node(node_to_keep, y=node_to_keep[0], x=node_to_keep[1])
                    G_simplified.add_edge(prev_node_kept, node_to_keep, length=cumul_dist)

                    # Reset
                    cumul_dist = 0
                    last_kept_idx = i

            # Connect the last segment
            end_node = path[-1]
            prev_node_kept = path[last_kept_idx]
            G_simplified.add_edge(prev_node_kept, end_node, length=cumul_dist)

    # import matplotlib.pyplot as plt  
    # pos0 = {n: (n[1], -n[0]) for n in G.nodes} # x, -y for image display
    # pos1 = {n: (n[1], -n[0]) for n in G_simplified.nodes} # x, -y for image display
    # plt.figure()
    # nx.draw(G, pos0, node_size=5, node_color='blue', edge_color='lightgray', with_labels=False)
    # nx.draw(G_simplified, pos1, node_size=15, node_color='green', edge_color='orange', with_labels=False)
    # plt.axis('equal')
    # plt.show()
    return G_simplified
