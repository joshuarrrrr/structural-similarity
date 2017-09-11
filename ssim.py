#!/usr/bin/python3

import argparse
import numpy as np
from numpy.lib.arraypad import _validate_lengths
from PIL import Image
from scipy.ndimage import gaussian_filter


def crop(ar, crop_width, copy=False, order='K'):
    '''Crop numpy array at the borders by crop_width.

    Source: www.github.com/scikit-image.'''

    ar = np.array(ar, copy=False)
    crops = _validate_lengths(ar, crop_width)
    slices = [slice(a, ar.shape[i] - b) for i, (a, b) in enumerate(crops)]

    if copy:
        cropped = np.array(ar[slices], order=order, copy=True)
    else:
        cropped = ar[slices]
    return cropped


def compute_ssim(x, y, win_size=11, sigma=1.5, L=255, K1=0.01, K2=0.03):
    '''Computes the structural similarity index as described by Wang et al.

    Parameters:
    x: image to compute the SSIM for
    y: reference image
    win_size: size of the sliding window (in pixels)
    sigma: sigma used for gaussian filter
    L: dynamic range of the image (defaults to 255 for 8-bit)
    K1, K2: constants for numerical stability
        (default to values used by Wang et al)

    Returns:
    mssim: mean structural similarity index
        (float between -1 and 1 where 1 means the images match perfectly)
    ssim: structural similarity index map for the full image
    '''

    # constants to avoid numerical instabilities close to zero
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2

    # compute weighted means
    ux = gaussian_filter(x, sigma)
    uy = gaussian_filter(y, sigma)

    # compute weighted variances
    uxx = gaussian_filter(x * x, sigma)
    uyy = gaussian_filter(y * y, sigma)
    uxy = gaussian_filter(x * y, sigma)

    # compute weighted covariances
    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    # compute the terms of eq. 13 of Wang et al
    ssim = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux**2 + uy**2 + C1) *
                                                    (vx + vy + C2))

    # compute mean SSIM not for the border areas where no full sliding window
    # was applied
    pad = (win_size - 1) // 2
    mssim = crop(ssim, pad).mean()

    # return mean SSIM as well as the full map
    return mssim, ssim


def main():
    parser = argparse.ArgumentParser(
        description='Compute the Structural Similarity Index of two images.')
    parser.add_argument('image', help='Filename of the image to test')
    parser.add_argument('ref_image', help='Filename of the reference image')
    parser.add_argument(
        '-p',
        '--plot',
        help='plot the SSIM for the image',
        action='store_true')
    args = parser.parse_args()

    image = Image.open(args.image).convert('L')
    ref_image = Image.open(args.ref_image).convert('L')

    if not image.size == ref_image.size:
        print('Error: The two images need to have the same size!')
        return

    image = np.array(image).astype(np.float64)
    ref_image = np.array(ref_image).astype(np.float64)

    mssim, ssim = compute_ssim(image, ref_image)
    print(mssim)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.imshow(ssim, cmap='magma')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    main()
