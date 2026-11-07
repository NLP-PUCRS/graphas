"""Structural feature extraction from a word graph.

This module owns the *measurement* stage of the pipeline: given a built
``networkx.MultiDiGraph``, it computes the dictionary of structural features that
quantify how organized or fragmented the underlying discourse is. Graph
*construction* (text -> graph) lives in :mod:`graphas.graphs.graph`; this
module only reads a finished graph and never builds one.

Keeping measurement separate from construction is what lets the sliding-window
mode reuse the exact same feature extractor over each window without dragging in
the text-to-graph modes.
"""

from collections import Counter

import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities, modularity

# features of the result that are node SETS, not scalars (their sizes are the
# scalar LCC / LSC features). Declared here — the module that PRODUCES them —
# and imported wherever the result is aggregated or flattened.
_SET_FEATURES = ('sLCC', 'sLSC')


class _graphStatistics():
    """Compute the structural feature dictionary of a ``MultiDiGraph``."""

    def __init__(self, graph):
        self.graph = graph

    def statistics(self):
        """Return the dictionary of structural features of the graph."""
        res = dict()
        graph = self.graph
        res['N'] = graph.number_of_nodes()
        res['E'] = graph.number_of_edges()

        # --- edge multiplicities (directed, parallel same-direction edges) -----
        edge_counts = Counter(graph.edges())
        # RE (Repeated Edges): extra same-direction edges beyond the first one.
        res['RE'] = int(sum(c - 1 for c in edge_counts.values() if c > 1))
        # PE (Parallel Edges): unordered pairs connected in BOTH directions
        # (u->v AND v->u).
        directed = set(edge_counts)
        res['PE'] = int(sum(1 for (u, v) in directed
                            if u != v and (v, u) in directed) // 2)

        # --- connectivity ------------------------------------------------------
        res['CC'] = nx.number_weakly_connected_components(graph)
        # sets with the largest number of nodes over all connected components
        res['sLCC'] = max(nx.weakly_connected_components(graph), key=len)
        res['sLSC'] = max(nx.strongly_connected_components(graph), key=len)
        # number of nodes in the largest set
        res['LCC'] = len(res['sLCC'])
        res['LSC'] = len(res['sLSC'])

        # --- extended static features ------------------------------------------
        # ATD (Average Total Degree): mean of in-degree + out-degree per node,
        # counting parallel edges — equals 2E/N.
        res['ATD'] = round(2 * res['E'] / res['N'], 4)
        # The projection features below are all computed on the UNDIRECTED simple
        # projection G_u (parallel edges collapse, direction is ignored); the
        # recurrence features above stay on the multigraph. One projection, built
        # once, shared by density / cohesion / heterogeneity / communities.
        gu = nx.Graph(graph)
        # Density: fraction of possible node pairs that are connected —
        # D = 2|E_u| / (N (N-1)), with self-loops excluded from E_u: a
        # self-loop connects no pair of distinct nodes (self-repetition is
        # measured by L1), and excluding it keeps D within [0, 1].
        n = gu.number_of_nodes()
        e = gu.number_of_edges() - nx.number_of_selfloops(gu)
        res['density'] = round(2 * e / (n * (n - 1)), 4) if n > 1 else 0.0
        # Diameter / ASP need finite shortest paths, so both are measured on the
        # UNDIRECTED simple view of the largest weakly connected component (the
        # largest subgraph on which every pairwise distance is finite).
        lcc = nx.Graph(graph.subgraph(res['sLCC']))
        if lcc.number_of_nodes() > 1:
            # Diameter: longest shortest path between any two nodes of the LCC.
            res['diameter'] = nx.diameter(lcc)
            # ASP: average shortest path length over all node pairs of the LCC.
            res['ASP'] = round(nx.average_shortest_path_length(lcc), 4)
        else:
            # degenerate LCC (a single node): no path exists to measure
            res['diameter'] = 0
            res['ASP'] = 0.0
        # LC (Local Cohesion, average clustering coefficient): fraction of each
        # node's neighbors that are themselves connected, averaged over all
        # nodes. Defined on the undirected simple view (parallel edges and
        # direction do not add triangles).
        res['LC'] = round(nx.average_clustering(gu), 4)
        # CH (Connectivity Heterogeneity, Shannon entropy of the degree
        # distribution of G_u) — the heterogeneity of the connection structure.
        # Repetition intensity is deliberately NOT part of this signal (it is
        # measured by RE / L1-L3 / ATD on the multigraph): one metric, one signal.
        _, counts = np.unique([d for _, d in gu.degree()], return_counts=True)
        p = counts / counts.sum()
        # `+ 0.0` normalizes the -0.0 a single-degree graph would otherwise yield
        res['CH'] = round(float(-(p * np.log2(p)).sum()) + 0.0, 4)
        # Community structure of G_u: CS (Community Strength, modularity of
        # the detected partition) and WC (Word Communities, the number of
        # communities found). Clauset-Newman-Moore greedy modularity: fully
        # deterministic (no seed) and always terminates. Louvain was dropped:
        # its partition depends on the seed, and networkx's implementation
        # (3.6.1) loops forever on some small window graphs.
        if gu.number_of_edges():
            communities = greedy_modularity_communities(gu)
            res['CS'] = round(float(modularity(gu, communities)), 4)
            res['WC'] = len(communities)
        else:
            # edgeless graph: every node is its own community, modularity is
            # undefined (no edges to score) and reported as 0.0
            res['CS'] = 0.0
            res['WC'] = gu.number_of_nodes()

        # --- loops via traces of A, A^2, A^3 -----------------------------------
        # Use the SPARSE adjacency matrix and sparse matmul (cost ~ number of
        # edges), so it scales to long texts and to the many windows of the
        # sliding-window mode.
        adj = nx.adjacency_matrix(graph)          # scipy sparse; parallel edges summed
        adj2 = adj @ adj
        adj3 = adj2 @ adj
        res['L1'] = int(adj.diagonal().sum())
        res['L2'] = int(adj2.diagonal().sum())
        res['L3'] = int(adj3.diagonal().sum())
        return res
