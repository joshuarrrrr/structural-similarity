# Structural Similarity Index
This is a python implementation of the structural similarity approach to assess image quality.
It is based on "Image quality assessment: from error visibility to structural similarity." by Wang et al. and the implementation of scikit-image.

# Usage
The script takes the paths to the image file and the reference image file.
It optionally plots the computed structural similarity map.
For usage see `./ssim.py -h`

## Requirements
The script requires the following python libraries:
- [pillow](https://python-pillow.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [matplotlib](https://matplotlib.org/) (optional, for plotting)
