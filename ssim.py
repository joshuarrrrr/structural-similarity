#!/usr/bin/python3

import argparse
from PIL import Image


def ssim(image, ref_image):
    return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Compute the Structural Similarity Index of two images.')
    parser.add_argument('image')
    parser.add_argument('ref_image')
    args = parser.parse_args()

    image = Image.open(args.image)
    ref_image = Image.open(args.ref_image)

    if not image.size == ref_image.size:
        print('Error: The two images need to have the same size!')
        return

    print(ssim(image, ref_image))


if __name__ == '__main__':
    main()
