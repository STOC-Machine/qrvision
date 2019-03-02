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
The vertical (first) index is the magnitude of that vector rho.
The horizontal (second) index is the polar angle of the vector theta, clockwise from the horizontal.

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
    max_rho = image.shape[0] + image.shape[1]
    
    iterator = np.nditer(image, flags=['multi_index'])
    while(not iterator.finished):
        if(iterator[0] != 0):
            for i in range(0, theta_buckets):
                theta = np.pi * (1.0 * i / theta_buckets)
                rho = (iterator.multi_index[1] * math.cos(theta)) + (iterator.multi_index[0] * math.sin(theta))
                j = round((rho + max_rho) / (2 * max_rho / (1.0 * rho_buckets)))
                accum[j][i] += 1
        iterator.iternext()
    return accum

def get_x(rho, theta, y):
    return (rho / np.cos(theta)) - (y * np.tan(theta))

def get_accum_endpoints(acc, peak, dim):
    rho = hough_parallelogram.convert_rho(peak[0], max_rho, 500)
    theta = hough_parallelogram.convert_angle(peak[1], 100)

    if theta == np.pi / 2:
        left = (0, int(rho))
        right = (dim[1], int(rho))
        return left, right

    x_top = int(get_x(rho, theta, 0))
    x_bottom = int(get_x(rho, theta, dim[0]))

    top = (x_top, 0)
    bottom = (x_bottom, dim[0])
    return top, bottom

max_rho = 0
rho_buckets = 500
theta_buckets = 100

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
    accumulated = accumulate(edges, theta_buckets, rho_buckets)
    maximum = np.amax(accumulated)
    
    # Run accumulator
    accumimage = (255.0 / maximum) * accumulated
    cv2.imshow("Accum", accumulated.astype(np.uint8))
    cv2.imwrite("accum.jpg", accumimage.astype(np.uint8))

    # Run enhancement
    enhanced = hough_parallelogram.enhance(accumulated, 20, 20)
    enhanceimage = (255.0 / np.amax(enhanced)) * enhanced
    cv2.imshow("Enhanced", enhanceimage.astype(np.uint8))
    cv2.imwrite("enhaced.jpg", enhanceimage.astype(np.uint8))
    print("Max element in enhanced: {}".format(np.amax(enhanced)))
    
    # Test findPeaks
    max_rho = img.shape[0] + img.shape[1]
    peaks = hough_parallelogram.findPeaks(enhanced, 500)
    print("Number of peaks: {}".format(len(peaks)))
    for peak in peaks:
        rho = peak[0]
        theta = peak[1]
        print("Peak: rho {}, theta {}, height {}".format(rho, theta, accumulated[rho][theta]))

        top_point, bottom_point = get_accum_endpoints(accumulated, peak, edges.shape)
        cv2.line(edges, top_point, bottom_point, (255, 0, 0))
    cv2.imshow("Edges with lines", edges)

    # Test findParallelograms
    peak_pairs = hough_parallelogram.findPeakPairs(peaks, accumulated, 3.0, 0.3, max_rho, rho_buckets, theta_buckets)
    print("Number of peak pairs: {}".format(len(peak_pairs)))
    parallelograms = hough_parallelogram.findParallelograms(peak_pairs, accumulated, 0.7)
    print("Number of parallelograms: {}".format(len(parallelograms)))
    for parallelogram in parallelograms:
        rho0 = parallelogram[0][0]
        theta0 = parallelogram[0][1]
        rho1 = parallelogram[1][0]
        theta1 = parallelogram[1][1]
        # print("Peak 1: rho {}, theta {}, height {}".format(rho0, theta0, enhanced[rho][theta]))
        # print("Peak 2: rho {}, theta {}, height {}".format(rho1, theta1, enhanced[rho][theta]))
        # print(parallelogram)
        
        hough_parallelogram.find_parallelogram_vertices(parallelogram, max_rho, rho_buckets, theta_buckets)
    
    # Wait for keypress to continue, close old windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
