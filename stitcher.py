#!/usr/bin/python
import numpy as np
import cv2
from pyzbar.pyzbar import decode
from PIL import Image

'''
This file is stitch_cropped_fragments and its boilerplate code.
It takes the corners of the QR code and stitches them into a complete code
that should be readable.

You need to input the images in their correct corners.
Each corner also has to be cropped to just the QR code fragment.
Background increases the error in the reconstruction because it
takes everything at face value, it can't respond to any non-QR parts of the images.

The corner images should also be at the same scale and near the same size.
How exact that has to be is a matter of how forgiving the QR reader is;
we don't make any attempts to fix that here.
TODO: this could be a good thing to fix here.
If we do it elsewhere, I probably won't need to match the corner dimensions exactly,
although it may still be a good idea to double-check if it doesn't add much time.
'''

tl = cv2.imread('sample-images/top-left-small.png', 0)
tr = cv2.imread('sample-images/top-right-small.png', 0)
bl = cv2.imread('sample-images/bottom-left-small.png', 0)
br = cv2.imread('sample-images/bottom-right-small.png', 0)

tl = cv2.imread('sample-images/scaled-tl.png', 0)
tr = cv2.imread('sample-images/scaled-tr.png', 0)
bl = cv2.imread('sample-images/scaled-bl.png', 0)
br = cv2.imread('sample-images/scaled-br.png', 0)

def stitch_cropped_fragments(top_left, top_right, bottom_left, bottom_right):
    print('top left: {}'.format(top_left.shape))
    print('top right: {}'.format(top_right.shape))
    print('bottom left: {}'.format(bottom_left.shape))
    print('bottom right: {}'.format(bottom_right.shape))

    top_left, top_right, bottom_left, bottom_right = scale_match_images(top_left, top_right, bottom_left, bottom_right)

    # Stitch sides together
    cropped = remove_blocks_from_fragment(top_right, 11, axis=1)
    top = np.concatenate((top_left, cropped), axis=1)
    cropped = remove_blocks_from_fragment(bottom_right, 11, axis=1)
    bottom = np.concatenate((bottom_left, cropped), axis=1)

    # Stitch top and bottom together
    cropped = remove_blocks_from_fragment(bottom, 11, axis=0)
    recon = np.concatenate((top, cropped), axis=0)
    return recon

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

def get_max_dimensions(*images):
    # Get dimensions
    max_width = -float('inf')
    max_height = -float('inf')
    for image in images:
        width = image.shape[1]
        if width > max_width:
            max_width = width
        height = image.shape[0]
        if height > max_height:
            max_height = height
    assert(max_width != float('inf') and max_height != float('inf'))
    assert(max_width > 0 and max_height > 0)
    return (max_height, max_width)

def scale_match_images(top_left, top_right, bottom_left, bottom_right, scale_down=True):
    # Crop images to square: should be roughly square to begin
    top_left, top_right, bottom_left, bottom_right = crop_images_to_square(top_left, top_right, bottom_left, bottom_right)

    # Get min/max size
    if scale_down:
        scale_size = get_min_dimensions(top_left, top_right, bottom_left, bottom_right)
    else:
        scale_size = get_max_dimensions(top_left, top_right, bottom_left, bottom_right)
    print('Scaled size: {}'.format(scale_size))

    # scale images to scale_size
    top_left = cv2.resize(top_left, scale_size)
    top_right = cv2.resize(top_right, scale_size)
    bottom_left = cv2.resize(bottom_left, scale_size)
    bottom_right = cv2.resize(bottom_right, scale_size)
    assert top_left.shape == top_right.shape == bottom_left.shape == bottom_right.shape
    print("Scales match!")

    return top_left, top_right, bottom_left, bottom_right

def crop_images_to_square(top_left, top_right, bottom_left, bottom_right):
    # crop top left
    square_size = min(top_left.shape)
    top_left = crop_edge(top_left, 'bottom', top_left.shape[0] - square_size)
    top_left = crop_edge(top_left, 'right', top_left.shape[1] - square_size)
    assert top_left.shape[1] == top_left.shape[0]

    # crop top right
    square_size = min(top_right.shape)
    top_right = crop_edge(top_right, 'bottom', top_right.shape[0] - square_size)
    top_right = crop_edge(top_right, 'left', top_right.shape[1] - square_size)
    assert top_right.shape[1] == top_right.shape[0]

    # crop bottom left
    square_size = min(bottom_left.shape)
    bottom_left = crop_edge(bottom_left, 'top', bottom_left.shape[0] - square_size)
    bottom_left = crop_edge(bottom_left, 'right', bottom_left.shape[1] - square_size)
    assert bottom_left.shape[1] == bottom_left.shape[0]

    # crop bottom left
    square_size = min(bottom_right.shape)
    bottom_right = crop_edge(bottom_right, 'top', bottom_right.shape[0] - square_size)
    bottom_right = crop_edge(bottom_right, 'left', bottom_right.shape[1] - square_size)
    assert bottom_right.shape[1] == bottom_right.shape[0]

    return top_left, top_right, bottom_left, bottom_right

def remove_blocks_from_fragment(fragment, fragment_size_blocks, axis):
    width_px = fragment.shape[axis]
    block_width = int(width_px / fragment_size_blocks)
    return np.delete(fragment, np.s_[0:block_width], axis)

def crop_edge(image, edge, cropped_px):
    if edge == 'top':
        # Vertical axis: 0; delete pixels before cropped_px
        image = np.delete(image, np.s_[0:cropped_px], 0)
    elif edge == 'bottom':
        # Vertical axis: 0; delete pixels before height - cropped_px
        image = np.delete(image, np.s_[image.shape[0] - cropped_px:], 0)
    elif edge == 'left':
        # Horizontal axis: 1; delete pixels before cropped_px
        image = np.delete(image, np.s_[0:cropped_px], 1)
    elif edge == 'right':
        # Horizontal axis: 1; delete pixels after width - cropped_px
        image = np.delete(image, np.s_[image.shape[1] - cropped_px:], 1)
    return image

reconstruction = stitch_cropped_fragments(tl, tr, bl, br)
cv2.imshow('Recon', reconstruction)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Test that the QR code reader can read this
decoded_base_object = decode(Image.open('sample-images/qr-base.png'))
decoded_base = int(decoded_base_object[0].data)
assert decoded_base == 2468, 'Base QR Code not readable: {}'.format(decoded_base)
decoded_recon = int(decode(reconstruction)[0].data)
assert decoded_base == decoded_recon, 'Reconstruction not readable: sould be {}, returns {}'.format(decoded_base, decoded_recon)
