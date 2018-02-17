#!/usr/bin/env python3
"""
...

Synopsis: ...
"""

import argparse
import sys

from PIL import Image, ImageDraw
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

def create_thumbnail(image, thumbnail_max = 100):
    thumbnail = image.copy()
    thumbnail.thumbnail((thumbnail_max, thumbnail_max), Image.ANTIALIAS)
    return thumbnail

def prominent_colors(image, num_colors):
    thumbnail = create_thumbnail(image)
    vertices = vertices_from_image(thumbnail)

    # Because the vertices are colors they should all form finite
    # numbers, so we can disable check_finite
    num_clusters = num_colors
    (centroid_codebook, _) = kmeans(vertices, num_clusters, check_finite = False)
    (codes, _) = vq(vertices, centroid_codebook, check_finite = False)
    (counts, bins) = histogram(codes, len(centroid_codebook))
    most_frequent = argsort(counts)[::-1]

    centroid_codebook = centroid_codebook.astype(int)
    return [tuple(centroid_codebook[most_frequent[i]]) for i in range(num_colors)]

def add_color_range(image, colors, size = 50):
    draw = ImageDraw.Draw(image)
    for (range_idx, color) in enumerate(colors):
        draw.rectangle(
            xy=((0, range_idx * size), (size, (range_idx + 1) * size)),
            fill=color, outline=None)

def pta(image_arg):
    image = Image.open(image_arg)

    colors = prominent_colors(image, 5)
    print('The most prominent colors are:\n%s' % '\n'.join(
        ['0x%02X%02X%02X' % color for color in colors]))

    add_color_range(image, colors)
    if image_arg is sys.stdin:
        image.save(sys.stdout, 'PNG')
    else:
        image.save('out.png', 'PNG')

def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--image', type=argparse.FileType('rb'), default=sys.stdin,
        help=("..."))
    return parser

if __name__ == "__main__":
    args = parser().parse_args()
    with args.image:
        pta(args.image)
