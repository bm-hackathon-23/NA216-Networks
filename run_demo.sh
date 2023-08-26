#!/bin/bash

## convert labels to fMRI resolution
#mri_convert -i Label052_005.nii.gz --like fmri_aneth_raw_005.nii.gz -rt nearest -o test.nii.gz

## run the example nigsp command
nigsp -f ./fmri_aneth_raw_005.nii.gz \
      -s ./FixedSC_005.csv \
      -sdi -dfc \
      -a ./test.nii.gz \
      -odir ./demo_results

## the default output extension isn't parsed in main?
