#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path
import nibabel as nib
from nibabel.filebasedimages import ImageFileError
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import cm

# Command line interface
parse = ArgumentParser()
parse.add_argument('-d', '--dir', dest='data_pth', type=str, help='The relative path to the directory containing the data.')
parse.add_argument('--viz', dest='plot', action='store_true', help='A flag indicating whether to display the graphs for all labels.')
parse.add_argument('-2d', dest='two_d', action='store_true', help='A flag indicating whether the display should be in 2D or 3D.')
args = parse.parse_args()

# Iterate over all available data file
data_path = Path(args.data_path).expanduser().resolve()
for f in data_path.iterdir():
    if f.is_file():
        try:
            data = nib.load(f)
        except ImageFileError:
            print(f'Error loading file: {f}')
            continue
        header = data.header
        print('#'*100)
        print(header)

        # If visualization is required display content of nifti file
        if args.plot:
            im = data.get_fdata()

            # 2D data plot
            if args.two_d:
                fig, ax = plt.subplots()
                ax.imshow(im[:, :, 0])

                # Slider across slices
                ax_slice = fig.add_axes([0.2, 0, 0.65, 0.03])
                slider = Slider(ax=ax_slice,
                                valstep=list(range(im.shape[2])),
                                valinit=0,
                                valmin=0,
                                valmax=im.shape[2],
                                label='slice')
                def update(val):
                    ax.imshow(im[:, :, int(val)])
                    fig.canvas.draw_idle()

                slider.on_changed(update)

            # Default to 3D data plot
            else:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                im = np.where(im < 0.08, False, True)
                ax.voxels(im[:, :, :, 0])

                # Slider across time
                ax_slice = fig.add_axes([0.2, 0, 0.65, 0.03])
                slider = Slider(ax=ax_slice,
                                valstep=list(range(im.shape[-1])),
                                valinit=0,
                                valmin=0,
                                valmax=im.shape[-1],
                                label='slice')
                def update(val):
                    ax.voxels(im[:, :, :, int(val)])
                    fig.canvas.draw_idle()

                slider.on_changed(update)

            # Display the graph
            plt.show()
