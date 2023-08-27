#!/usr/bin/env python3
from pathlib import Path
from nilearn import image
from nilearn.decomposition import CanICA, DictLearning
from nilearn.masking import compute_brain_mask, compute_epi_mask
from nilearn.plotting import plot_prob_atlas, plot_stat_map
from matplotlib import pyplot as plt

ROOT = Path('Data', 'MRI-NA216', 'invivo', 'i_fMRI_Aneth_raw').expanduser().resolve()
avg_img = image.load_img(Path('Data', 'MRI-NA216', 'invivo', 'i_Average', 'Average_T1WI.nii.gz'))

# Compute brain mask
mask = compute_brain_mask(target_img=avg_img,
                          threshold=0.1,
                          connected=False,
                          opening=False,
                          memory='nilearn_cache',
                          mask_type='gm')

# Compute CanICA
can_ica = CanICA(mask=mask,
                 n_components=15,
                 smoothing_fwhm=2.0,
                 memory='nilearn_cache',
                 memory_level=2,
                 verbose=0,
                 random_state=0,
                 standardize='zscore_sample',
                 n_jobs=4)
can_ica.fit(avg_img)

# Visualize CanICA
ica_comps_img = can_ica.components_img_
plot_prob_atlas(ica_comps_img, title='All ICA components')

# Compute DictLearn
dict_lrn = DictLearning(mask=mask,
                        n_components=15,
                        memory='nilearn_cache',
                        memory_level=2,
                        verbose=0,
                        random_state=0,
                        n_epochs=1,
                        smoothing_fwhm=2.0,
                        standardize='zscore_sample',
                        alpha=2)
dict_lrn.fit(avg_img)

# Visualize DictLearn
dict_lrn_comps_img = dict_lrn.components_img_
plot_prob_atlas(dict_lrn_comps_img, title='All DictLearn components')

# Show all plots
plt.show()
