import cv2
import glob
import numpy as np
import sys
import math
import hough_parallelogram

"""
Generates the Hough accumulator array for the given image.
The lines are defined by the point in the image (ie, the pixel indices of the image)
and the angle from the vertical, clockwise.

The array is in a polar coordinate system based on the normal vector to the line.
The vertical (first) index is the magnitude of that vector.
The horizontal (second) index is the polar angle of the vector, clockwise from the horizontal.

TODO: change angle from going up to 2*pi. A line at theta=pi/2 is the same as one at 3*pi/2.

TODO: double check how the angle is handled. I don't know for sure if this works as expected.
According to the openCV spec (which I think we should stick to), if the line passes below the
origin (when it intersects the vertical axis), they take the radius as positive. If it passes
above the origin, the radius is negative.
This is definitely not implemented correctly here. We don't have negative radius and angle goes
0 to 2*pi, unlike theirs.
Alternatively, we could define the angle from 0 to 2*pi, instead of pi. It would differ from
their spec, but as long as we're consistent and know that, we should be ok.
"""
def accumulate(image, theta_buckets, rho_buckets):
    
    accum = np.zeros((rho_buckets, theta_buckets))
    
    max_rho = math.sqrt((image.shape[0] * image.shape[0]) + (image.shape[1] * image.shape[1]))
    
    iterator = np.nditer(image, flags=['multi_index'])
    while(not iterator.finished):
        if(iterator[0] != 0):
            print(iterator.multi_index)
            for i in range(0, theta_buckets):
                theta = (2 * np.pi * i) / (1.0 * theta_buckets)
                rho = (iterator.multi_index[1] * math.cos(theta)) + (iterator.multi_index[0] * math.sin(theta))
                j = int((rho + max_rho) / (2 * max_rho / (1.0 * rho_buckets)))
                accum[j][i] += 1
        iterator.iternext()
    return accum

files = glob.glob(sys.argv[1])
while len(files) > 0:
    file = files.pop(0)
    img = cv2.imread(file)
    
    if img is None:
        print('Error: could not read image file {}, skipping.'.format(file))
        continue
    
    cv2.imshow("Original", img)
    edges = cv2.Canny(img, 100, 100)
    cv2.imshow("Edges", edges)
    accumulated = accumulate(edges, 50, 50)
    maximum = np.amax(accumulated)
    
    # Run accumulator
    accumimage = (255.0 / maximum) * accumulated
    cv2.imshow("Accum", accumulated.astype(np.uint8))
    enhanced = hough_parallelogram.enhance(accumulated, 10, 10)
    cv2.imshow("Enhanced", enhanced.astype(np.uint8))
    
    # Wait for keypress to continue, close old windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
