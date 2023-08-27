#!/usr/bin/env python3
from pathlib import Path
import networkx as nx
import clustering_nilearn as cn
import measure_complexities_over_segment_maps as mcosm
from matplotlib import pyplot as plt

NB_JOBS = 4

def main(nb_clusters):
# Get the parcellation facility
    parc = cn.init_parc('ward', nb_clusters, NB_JOBS)
    area_segments = cn.get_region_signals(parc)

# Load the average image
    data = Path('Data', 'MRI-NA216', 'invivo', 'i_Average', 'Average_T1WI.nii.gz').resolve()

# Create area labels
    area_labels = [f'area_{i}' for i in range(nb_clusters)]

# Extract the time series from the data
    time_series = mcosm.make_time_series(data, area_segments)

# Compute the correlation matrix based on the time series
    corr_mat = mcosm.make_correlation_matrix(time_series, area_labels, viz_needed=False)

# Finally, build a graph using the correlation matrix as adjacency matrix
    graph = mcosm.make_graph(corr_mat, area_labels, viz_needed=False)
    print(graph.nodes)


if __name__ == '__main__':
    for nb_clusters in [104, 52, 26, 13]:
        main(nb_clusters)
        exit()
