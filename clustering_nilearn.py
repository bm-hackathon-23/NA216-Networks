#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, ticker
from nilearn import datasets, plotting, image
from nilearn.image import get_data, index_img, mean_img
from nilearn.regions import Parcellations, signals_to_img_labels

ROOT = Path('data', 'i_fMRI_Aneth_raw').expanduser().resolve()
#ROOT = Path('Data', 'MRI-NA216', 'invivo', 'i_fMRI_Aneth_raw').resolve()

def init_parc(name, nb_clusters, nb_jobs):
    if name == 'ward':
        return Parcellations(
            method='ward',
            n_parcels=nb_clusters,
            standardize=False,
            smoothing_fwhm=2.0,
            memory_level=1,
            memory='nilearn_cache',
            verbose=0,
            n_jobs=nb_jobs)
    elif name == 'kmeans':
        return Parcellations(
            method='kmeans',
            n_parcels=args.nb_clusters,
            standardize='zscore_sample',
            smoothing_fwhm=4.0,
            memory_level=1,
            memory='nilearn_cache',
            verbose=0,
            n_jobs=args.nb_jobs)
    elif name == 'hkmeans':
        return Parcellations(
            method='hierarchical_kmeans',
            n_parcels=args.nb_clusters,
            standardize='zscore_sample',
            smoothing_fwhm=4.0,
            memory_level=1,
            memory='nilearn_cache',
            verbose=0,
            n_jobs=args.nb_jobs)


def get_roi(parc):
    # Fit the image
    img = image.load_img(Path('Data', 'MRI-NA216', 'invivo', 'i_Average', 'Average_T1WI.nii.gz'))
    parc.fit(img)
    return parc.labels_img_


def plot_roi(labels_img):
    # Visualizing the clusters
    first_plot = plotting.plot_roi(
        labels_img, title='Parcellation', display_mode='xz'
    )
    plt.show()


def get_region_signals(parc):

    # Load average image
    avg_img = image.load_img(ROOT.parent.joinpath('Average_T1WI.nii.gz'))
    #avg_img = image.load_img(ROOT.parent.joinpath('i_Average', 'Average_T1WI.nii.gz'))
    parc.fit(avg_img)

    # Transform images
    reg_sigs = []
    for f in ROOT.iterdir():
        if f.is_file():
            # Load list of images
            img = image.load_img(f)

            # Transform the image
            try:
                sig = parc.transform(img)
            except:
                print('Error')
                img.uncache()
                continue

            reg_sigs.append(sig)
            img.uncache()

    shape = None
    cleaned_reg_sigs = []
    for sig in reg_sigs:
        if shape is None:
            shape = sig.shape
        elif sig.shape == shape:
            cleaned_reg_sigs.append(sig)


    reg_sigs = np.mean(np.concatenate(cleaned_reg_sigs), axis=0, keepdims=True)
    return signals_to_img_labels(reg_sigs, parc.labels_img_)


def save_img(img, file_path):
    img.to_filename(Path(file_path).resolve())


def plot_region_signals(reg_sigs):
    # Visualizing the clusters
    first_plot = plotting.plot_img(
        reg_sigs, title='ward parcellation', display_mode='xz'
    )
    plt.show()

if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--roi', dest='roi', action='store_true', help='A flag indicating whether to display the ROI or region signal (default).')
    parse.add_argument('--nb_clusters', dest='nb_clusters', type=int, help='The number of regions to define over the data.', required=True)
    parse.add_argument('--nb_jobs', dest='nb_jobs', type=int, help='The number of parallel jobs to execute at the same time.', default=1)
    ex_grp = parse.add_mutually_exclusive_group(required=True)
    ex_grp.add_argument('--ward', dest='ward', action='store_true')
    ex_grp.add_argument('--kmeans', dest='kmeans', action='store_true')
    ex_grp.add_argument('--hkmeans', dest='hkmeans', action='store_true')

    args = parse.parse_args()

    # Parcellation
    if args.ward:
        parc = init_parc('ward', args.nb_clusters, args.nb_jobs)
    elif args.kmeans:
        parc = init_parc('kmeans', args.nb_clusters, args.nb_jobs)
    elif args.hkmeans:
        parc = init_parc('hkmeans', args.nb_clusters, args.nb_jobs)

    # Visualization
    if args.roi:
        roi = get_roi(parc)
        save_img(roi, 'region_of_interest.nii.gz')
        plot_roi(roi)
    else:
        reg_sig = get_region_signals(parc)
        save_img(reg_sig, 'region_signals.nii.gz')
        plot_region_signals(reg_sig)
