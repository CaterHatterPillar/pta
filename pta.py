#!/usr/bin/env python3
"""
...

Synopsis: ...
"""

import argparse
import sys

from PIL import Image

from numpy import argsort, asarray
from scipy import histogram, product
from scipy.cluster.vq import kmeans, vq

# scipy only works on floaing point numbers
def vertices_from_image(image):
    vertices = asarray(image)
    num_channels = vertices.shape[2]
    resolution = vertices.shape[:2]
    num_vertices = product(resolution)
    return vertices.reshape(num_vertices, num_channels).astype(float)

THUMBNAIL_MAX = 100
def prominent_colors(image, num_colors):
    image.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX), Image.ANTIALIAS)
    vertices = vertices_from_image(image)

    # Because the vertices are colors they should all form finite
    # numbers, so we can disable check_finite
    num_clusters = num_colors
    (centroid_codebook, _) = kmeans(vertices, num_clusters, check_finite = False)
    (codes, _) = vq(vertices, centroid_codebook, check_finite = False)
    (counts, bins) = histogram(codes, len(centroid_codebook))
    most_frequent = argsort(counts)[::-1]

    centroid_codebook = centroid_codebook.astype(int)
    return [tuple(centroid_codebook[most_frequent[i]]) for i in range(num_colors)]

def pta(image):
    colors = prominent_colors(image, 5)
    print('The most prominent colors are:\n%s' % '\n'.join(
        ['0x%02X%02X%02X' % color for color in colors]))

def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--image', type=argparse.FileType('rb'), default=sys.stdin,
        help=("..."))
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    with args.image:
        pta(Image.open(args.image))
