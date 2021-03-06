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
    # rho = hough_parallelogram.convert_rho(peak[0], max_rho, 500)
    # theta = hough_parallelogram.convert_angle(peak[1], 100)
    rho = peak[0]
    theta = peak[1]

    if theta == np.pi / 2:
        left = (0, int(rho))
        right = (dim[1], int(rho))
        return left, right

    x_top = int(get_x(rho, theta, 0))
    x_bottom = int(get_x(rho, theta, dim[0]))

    top = (x_top, 0)
    bottom = (x_bottom, dim[0])
    return top, bottom
    
def generate_color_spectrum(iteration):
    hue = 180 - (2 * iteration)
    color = np.uint8([[[hue, 255, 255]]])
    rgb = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
    return (np.asscalar(rgb[0]), np.asscalar(rgb[1]), np.asscalar(rgb[2]))

def ave_peak_height(pair):
    return (pair[0][2] + pair[1][2]) / 2.0

def draw_lines(image, accum, peaks, color, title):
    for peak in peaks:
        top_point, bottom_point = get_accum_endpoints(accum, peak, image.shape)
        # print("Point: {} {}".format(top_point, bottom_point))
        cv2.line(image, top_point, bottom_point, color)
    return image

def draw_pairs(image, accum, pairs):
    pairs.sort(key=ave_peak_height, reverse=True)
    iter = 0
    for pair in pairs:
        draw_image = np.zeros((image.shape[0], image.shape[1], 3))
        draw_image = draw_lines(draw_image, accum, pair, generate_color_spectrum(iter), "Pair 1")
        image = draw_lines(image, accum, pair, generate_color_spectrum(iter), "Pair 1")
        iter = iter + 1
        # cv2.destroyWindow("Pair {}".format(iter))
        # cv2.imshow("Pair 1", draw_image)
        # cv2.waitKey(0)
    cv2.imshow("Pairs", image)

def draw_parallelograms(image, accum, ps, title):
    iter = 0
    for parallelogram in ps:
        color = generate_color_spectrum(iter)
        iter = iter + 1
        for pair in parallelogram:
            image = draw_lines(image, accum, pair, color, "Pair 1")
    cv2.imshow(title, image)

def find_different_elements(one, two):
    if one == two:
        print("Agree")
        return

    if len(one) > len(two):
        print(two[len(one):])
        print("Number of disagreeing elements: {}".format(len(one) - len(two)))
        return
    if len(two) > len(one):
        print(one[len(two):])
        print("Number of disagreeing elements: {}".format(len(two) - len(one)))
        return
    different = 0
    for i in range(0, len(one)):
        if one[i] != two[i]:
            print("Disagreeing: {}".format(one[i]))
            print("             {}".format(two[i]))
            different += 1
    print("Number of disagreeing elements: {}".format(different))

max_rho = 0
rho_buckets = 500
theta_buckets = 500

files = glob.glob(sys.argv[1])
while len(files) > 0:
    file = files.pop(0)
    img = cv2.imread(file)
    
    if img is None:
        print('Error: could not read image file {}, skipping.'.format(file))
        continue
    
    # cv2.imshow("Original", img)
    edges = cv2.Canny(img, 100, 100)
    cv2.imshow("Edges", edges)
    accumulated = accumulate(edges, theta_buckets, rho_buckets)
    maximum = np.amax(accumulated)
    
    # Run accumulator
    accumimage = (255.0 / maximum) * accumulated
    # cv2.imshow("Accum", accumulated.astype(np.uint8))
    cv2.imwrite("accum.jpg", accumimage.astype(np.uint8))

    # Run enhancement
    enhanced = hough_parallelogram.enhance(accumulated, 20, 20)
    enhanceimage = (255.0 / np.amax(enhanced)) * enhanced
    # cv2.imshow("Enhanced", enhanceimage.astype(np.uint8))
    cv2.imwrite("enhaced.jpg", enhanceimage.astype(np.uint8))
    print("Max element in enhanced: {}".format(np.amax(enhanced)))
    
    # Test findPeaks
    max_rho = img.shape[0] + img.shape[1]
    peaks = hough_parallelogram.findPeaks(accumulated, enhanced, 500, max_rho, rho_buckets, theta_buckets)
    print("Number of peaks: {}".format(len(peaks)))
    
    # lines_image = np.zeros((edges.shape[0], edges.shape[1], 3))
    # draw_lines(lines_image, accumulated, peaks, (255, 0, 0), "Hough lines")

    peak_pairs = hough_parallelogram.findPeakPairs(peaks, accumulated, 3.0, 0.3, 0.3, max_rho, rho_buckets)
    print("Number of peak pairs: {}".format(len(peak_pairs)))
    
    parallelograms = hough_parallelogram.findParallelograms(peak_pairs, accumulated, 0.7, np.pi / 6)
    print("Number of parallelograms: {}".format(len(parallelograms)))

    valids = []
    all_errors = []
    for parallelogram in parallelograms:
        valid, error = hough_parallelogram.validate_parallelogram(parallelogram, edges, 0.6)

        if error != 1:
            all_errors.append([error, [parallelogram.pair_k.old_list_format(), parallelogram.pair_l.old_list_format()]])
            assert(type(all_errors[-1][0] == "float64"))
            if valid:
                valids.append([error, [parallelogram.pair_k.old_list_format(), parallelogram.pair_l.old_list_format()]])
                # cv2.waitKey(0)
    print("Number of valid parallelograms: {}".format(len(valids)))

    all_errors.sort(key=lambda x: x[0])
    for p in all_errors:
        test_image = np.zeros((edges.shape[0], edges.shape[1], 3))
        test_overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        # draw_parallelograms(test_image, accumulated, [p[1]], "Sorted parallelograms")
        # print("Agreement: {}".format(p[0]))
        # cv2.waitKey(0)
    # draw_parallelograms(test_image, accumulated, [p[1] for p in all_agreements], "Sorted parallelograms")

    # print(draw_parallelograms(lines_image, accumulated, parallelograms, "Parallelograms"))
    
    # Wait for keypress to continue, close old windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
