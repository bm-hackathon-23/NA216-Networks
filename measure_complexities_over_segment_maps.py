import os
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

from nilearn import plotting, image
from nilearn.signal import clean
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.regions import Parcellations, signals_to_img_labels

import clustering_nilearn
from pybdm import BDM

# goals:
# from a time series of the brain signals (average value over the area),
# get the underlying graph structure
# explore various area segmentations and the resulting graphs
# measure the complexity of the resulting graphs and pick the one with the lowest complexity

def plot_csv_matrix(data_file):

    # plot the aneth p correlation amtrix to compare with the one NiLearn makes
    aneth_correlations = np.genfromtxt(data_file, delimiter=',')
    plt.imshow(aneth_correlations)
    plt.colorbar()
    plt.show()

def make_time_series(data, labels):

    # create the masker for extracting time series
    masker = NiftiLabelsMasker(labels_img=labels, standardize=True)

    # extract the timeseries
    time_series = masker.fit_transform(data)

    # clean the time series
    clean_series = clean(time_series)

    # plot an example time series
    # plt.plot(np.arange(0, clean_series.shape[0]), clean_series[:, 0])
    # plt.show()

    return clean_series

def make_correlation_matrix(time_series, area_labels, id=0, parcellation=0, viz_needed=True):

    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    # Mask the main diagonal for visualization:
    np.fill_diagonal(correlation_matrix, 0)

    if viz_needed:
        # matrices are ordered for block-like representation
        #plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=area_labels, vmax=0.8, vmin=-0.8, reorder=True)
        #plt.savefig(os.path.join('figures', 'adj_matrix', str(parcellation) + '_' + str(id) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()

    return correlation_matrix

def make_graph(correlation_matrix, area_labels, id=0, parcellation=0, viz_needed=True):

    G = nx.from_numpy_array(correlation_matrix)

    # add node labels
    node_mapping = dict(zip(list(range(0, 104)), area_labels))
    G = nx.relabel_nodes(G, node_mapping)

    # see the distribution of weights to trim out the very small weights
    #plt.hist(correlation_matrix.flatten(), bins=100)
    #plt.show()
    # >>> for now, let's just keep everything greater than 0.1 or less than -0.1

    # filter out edges with small values
    threshold = 0.2
    edges_to_keep = list(filter(lambda x: G.get_edge_data(*x)['weight'] > threshold
                                          or G.get_edge_data(*x)['weight'] < -threshold, G.edges()))
    G = G.edge_subgraph(edges_to_keep).copy()

    if len(list(G.edges())) < 10:
        return None

    if viz_needed:
        # visualize the network
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        pos = nx.spring_layout(G)
        nx.draw(G, pos, node_size=50, node_color='b', edgelist=edges, edge_color=weights, width=2.0, alpha=0.7, edge_cmap=plt.cm.Blues)
        plt.savefig(os.path.join('figures', 'graphs', str(parcellation) + '_' + str(id) + '.png'))
        plt.clf()
        plt.cla()
        plt.close()

    return G

def measure_complexity(G):

    # binarize the adjacency matrix
    adj_matrix = nx.adjacency_matrix(G)
    adj_matrix = adj_matrix.toarray()
    adj_matrix = (adj_matrix > 0).astype(np.int_)

    # Initialize BDM object
    bdm = BDM(ndim=2)

    # Compute BDM
    k_complexity = bdm.bdm(adj_matrix)

    # BDM objects may also compute standard Shannon entropy in base 2
    entropy = bdm.ent(adj_matrix)

    return k_complexity, entropy

def meat_and_potatoes(id, parc_image, parcellation):

    # plot the given correlation matrix
    # aneth_correlations_file = os.path.join('data', 'i_AnethFC', 'AnethFC_005.csv')
    # plot_csv_matrix(aneth_correlations_file)
    # >>>>> It doesn't look the same at all, probably because this was made with a different pipeline

    # load in raw nifti file
    data = os.path.join('data', 'i_fMRI_aneth_raw', 'fmri_aneth_raw_' + id + '.nii.gz')

    if parc_image == 'given':

        # load in the pre-defined area segments
        parc_image = os.path.join('data', 'i_Label052', 'Label052_' + id + '.nii.gz')

    # load in the area labels
    labels_file = os.path.join('data', 'labels.txt')
    labels_file = open(labels_file, "r")
    area_labels = [x.strip() for x in labels_file]

    # make the time series
    time_series = make_time_series(data, parc_image)

    # turn that into a correlation matrix
    correlation_matrix = make_correlation_matrix(time_series, area_labels, id, parcellation)

    # turn that into a network by making the correlation matrix and adj matrix
    G = make_graph(correlation_matrix, area_labels, id, parcellation)

    # if the graph has less than 10 edges, throw it away
    if not G: return None, None
    if len(list(G.edges())) < 10: return None, None

    # if the graph has less than 10 edges, throw it away
    if len(list(G.edges())) < 10:
        return None, None

    # measure the complexity, binarize the weights
    k_complexity, entropy = measure_complexity(G)

    return k_complexity, entropy


if __name__ == '__main__':

    # get list of individual ids
    files = os.listdir(os.path.join('data', 'i_AnethFC'))
    if '.DS_Store' in files: files.remove('.DS_Store')  # osx.... >:c
    ids = list(map(lambda x: x.split('_')[-1].split('.')[0], files))[:2]

    # save the data in a list of dfs to concat for plotting
    dfs = []

    # loop over different parcellations
    parcellations = [13, 26, 'given']

    for parcellation in parcellations:

        # see if it is the default parcellation
        # if it is, then just use the given file
        if parcellation == 'given':
            parc_image = parcellation

        # else, we make our own
        else:
            parc = clustering_nilearn.init_parc('ward', parcellation, nb_jobs=1)

            # then get the region signals, which will return 3D image
            parc_image = clustering_nilearn.get_region_signals(parc)

        # then, loop over the individuals
        for id in ids:

            k_complexity, entropy = meat_and_potatoes(id, parc_image, parcellation)

            # remove graphs with no edges
            if not k_complexity:
                continue

            # save as df (for plotting
            data = {}
            data['k_complexity'] = [k_complexity]
            data['entropy'] = [entropy]
            data['id'] = [id]
            data['parcellation'] = [parcellation]
            df = pd.DataFrame.from_dict(data)
            dfs.append(df)

            print(id)

    # concat dfs
    df = pd.concat(dfs)

    # plot histogram
    g = sns.kdeplot(
        data=df, x="k_complexity", hue="parcellation",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0,
    )
    g.set_yscale("log")
    plt.savefig(os.path.join('figures', 'plots', 'k_complexity.png'))
    plt.clf()
    plt.cla()
    plt.close()

    sns.kdeplot(
        data=df, x="entropy", hue="parcellation",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0,
    )
    plt.savefig(os.path.join('figures', 'plots', 'entropy.png'))
    plt.clf()
    plt.cla()
    plt.close()

    # will want to compare these values to random graphs of the same size
