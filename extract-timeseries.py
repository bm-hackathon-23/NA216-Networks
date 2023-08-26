
import numpy as np

from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean

import matplotlib.pyplot as plt

#from nilearn.interfaces.fmriprep import load_confounds_strategy


## paths to files
bold = '/home/bcmcpher/Projects/mb-hackathon/nigsp/fmri_aneth_raw_005.nii.gz'
parc = '/home/bcmcpher/Projects/mb-hackathon/nigsp/Label052_005.nii.gz'

## load confounds (?)
#conf = load_confounds_strategy(bold, denoise_strategy='simple')

## create the masker for extracting time series
masker = NiftiLabelsMasker(labels_img=parc, standardize=True)

## extract the timeseries?
#time_series = masker.fit_transform(bold, confounds=conf[0])
time_series = masker.fit_transform(bold)

## clean the time series
clean_series = clean(time_series)

## plot an example time series
plt.plot(np.arange(0,clean_series.shape[0]), clean_series[:,0])
plt.show()
