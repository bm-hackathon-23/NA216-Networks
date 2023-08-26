import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from nilearn import plotting
from nilearn.signal import clean
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

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

    return clean_series

def make_correlation_matrix(time_series, area_labels):

    correlation_measure = ConnectivityMeasure(kind="correlation")
    correlation_matrix = correlation_measure.fit_transform([time_series])[0]

    # Mask the main diagonal for visualization:
    np.fill_diagonal(correlation_matrix, 0)

    # matrices are ordered for block-like representation
    #plotting.plot_matrix(correlation_matrix, figure=(10, 8), labels=area_labels, vmax=0.8, vmin=-0.8, reorder=True)
    #plt.show()

    return correlation_matrix


if __name__ == '__main__':

    # define the parameters
    individual_id = '005'

    # plot the given correlation matrix
    #aneth_correlations_file = os.path.join('data', 'i_AnethFC', 'AnethFC_005.csv')
    #plot_csv_matrix(aneth_correlations_file)

    # load in raw nifti file
    data = os.path.join('data', 'i_fMRI_aneth_raw', 'fmri_aneth_raw_005.nii.gz')

    # load in the pre-defined area segments
    area_segments = os.path.join('data', 'Label052_005.nii.gz')  # TODO: organize these files
    # TODO: remove all the data files I'm not suing

    # load in the area labels
    labels_file = os.path.join('data', 'labels.txt')
    labels_file = open(labels_file, "r")
    area_labels = [x.strip() for x in labels_file]

    # make the time series
    time_series = make_time_series(data, area_segments)

    # turn that into a correlation matrix
    correlation_matrix = make_correlation_matrix(time_series, area_labels)

    # turn that into a network by making the correlation matrix and adj matrix
    G = nx.from_numpy_array(correlation_matrix)

    # see the distribution of weights to trim out the very small weights



    # visualize the network

    # plot an example time series
    #plt.plot(np.arange(0, clean_series.shape[0]), clean_series[:, 0])
    #plt.show()
