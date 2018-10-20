#!/usr/bin/python
import numpy as np
import cv2
from pyzbar.pyzbar import decode
from PIL import Image

# Total thing: 21 blocks wide
# 10 each side, 1 shared in the middle

tl = cv2.imread('sample-images/top-left-small.png', 0)
tr = cv2.imread('sample-images/top-right-small.png', 0)
bl = cv2.imread('sample-images/bottom-left-small.png', 0)
br = cv2.imread('sample-images/bottom-right-small.png', 0)

def stitch_cropped_fragments(top_left, top_right, bottom_left, bottom_right):
    print('top left: {}'.format(top_left.shape))
    print('top right: {}'.format(top_right.shape))
    print('bottom left: {}'.format(bottom_left.shape))
    print('bottom right: {}'.format(bottom_right.shape))

    top_left, top_right, bottom_left, bottom_right = crop_images_to_size(top_left, top_right, bottom_left, bottom_right)

    # Stitch sides together
    cropped = remove_from_fragment(top_right, 11, axis=1)
    top = np.concatenate((top_left, cropped), axis=1)
    cropped = remove_from_fragment(bottom_right, 11, axis=1)
    bottom = np.concatenate((bottom_left, cropped), axis=1)
    cv2.imshow('Top', top)
    cv2.imshow('Bottom', bottom)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Stitch top and bottom together
    cropped = remove_from_fragment(bottom, 11, axis=0)
    recon = np.concatenate((top, cropped), axis=0)
    cv2.imshow('Complete', recon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def crop_images_to_size(top_left, top_right, bottom_left, bottom_right):
    min_dimensions = get_min_dimensions(top_left, top_right, bottom_left, bottom_right)
    print('Resizing images... dimensions: {}'.format(min_dimensions))
    min_width = min_dimensions[1]
    min_height = min_dimensions[0]

    # crop left images along right edge
    top_left = np.delete(top_left, np.s_[min_width:], 1)
    bottom_left = np.delete(bottom_left, np.s_[min_width:], 1)
    assert(top_left.shape[1] == bottom_left.shape[1])
    print('left width: {}'.format(top_left.shape[1]))

    # crop top images along bottom edge
    top_left = np.delete(top_left, np.s_[min_height:], 0)
    top_right = np.delete(top_right, np.s_[min_height:], 0)
    assert(top_left.shape[0] == top_right.shape[0])
    print('top height: {}'.format(top_left.shape[0]))

    # crop right images along left edge
    top_right = np.delete(top_right, np.s_[0:top_right.shape[1] - min_width], 1)
    bottom_right = np.delete(bottom_right, np.s_[0:bottom_right.shape[1] - min_width], 1)
    assert(top_right.shape[1] == bottom_right.shape[1])
    print('right width: {}'.format(top_right.shape[1]))

    # crop bottom images along top edge
    bottom_left = np.delete(bottom_left, np.s_[0:bottom_left.shape[0] - min_height], 0)
    bottom_right = np.delete(bottom_right, np.s_[0:bottom_right.shape[0] - min_height], 0)
    assert(bottom_left.shape[0] == bottom_right.shape[0])
    print('bottom height: {}'.format(bottom_left.shape[0]))

    return top_left, top_right, bottom_left, bottom_right

def get_min_dimensions(*images):
    # Get dimensions
    min_width = float('inf')
    min_height = float('inf')
    for image in images:
        width = image.shape[1]
        if width < min_width:
            min_width = width
        height = image.shape[0]
        if height < min_height:
            min_height = height
    assert(min_width != float('inf') and min_height != float('inf'))
    assert(min_width > 0 and min_height > 0)
    return (min_height, min_width)

def remove_from_fragment(fragment, width_blocks, axis):
    width_px = fragment.shape[axis]
    block_width = int(width_px / width_blocks)
    return np.delete(fragment, np.s_[0:block_width], axis)

stitch_cropped_fragments(tl, tr, bl, br)
# Test that the QR code reader can read this
decoded_base_object = decode(Image.open('sample-images/qr-base.png'))
decoded_base = int(decoded_base_object[0].data)
assert decoded_base == 2468, 'QR Code not readable: {}'.format(decoded_base)
