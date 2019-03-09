import numpy as np
import cv2

#section 1:find parallelograms in image
def enhance(acc, h, w):
    # Sums up h x w region around every pixel:
    # this is the "rectangular convolution" the paper mentioned.
    # This saves the sum result about each pixel to the matrix dst.
    convolution_matrix = np.ones((h, w))
    dst = cv2.filter2D(acc, -1, convolution_matrix)

    enhanced = np.zeros(acc.shape)
    # Now find the enhanced values at each pixel of the accumulator
    # using the convolved matrix
    # This is probably not the most efficient way, we could probably numpify it
    for rho in range(0, acc.shape[0]):
        for theta in range(0, acc.shape[1]):
            if dst[rho][theta] != 0:
                numerator = h * w * (acc[rho][theta]**2)
                enhanced[rho][theta] = (1.0 * numerator) / dst[rho][theta]
            # Otherwise, the value should be 0.
            # Since I initialized the enhanced array to zeros, do nothing.

    return enhanced

# Based on the OpenCV implementation: hough.cpp, line 84
def is_peak(acc, rho_index, theta_index, peak_thresh):
    if acc[rho_index, theta_index] < peak_thresh:
        return False

    # Check that acc[rho_index, theta_index] is greater than all neighboring values,
    # making sure that we don't go outside array bounds
    # Check vertical
    if rho_index > 0:
        if acc[rho_index - 1][theta_index] >= acc[rho_index][theta_index]:
            return False
    if rho_index < acc.shape[0] - 1:
        if acc[rho_index + 1][theta_index] >= acc[rho_index][theta_index]:
            return False
    # Check horizontal
    if theta_index > 0:
        if acc[rho_index][theta_index - 1] >= acc[rho_index][theta_index]:
            return False
    if theta_index < acc.shape[1] - 1:
        if acc[rho_index][theta_index + 1] >= acc[rho_index][theta_index]:
            return False
    # Check diagonal above
    if rho_index > 0 and theta_index > 0:
        if acc[rho_index - 1][theta_index - 1] >= acc[rho_index][theta_index]:
            return False
    if rho_index > 0 and theta_index < acc.shape[1] - 1:
        if acc[rho_index - 1][theta_index + 1] >= acc[rho_index][theta_index]:
            return False
    # Check diagonal below
    if rho_index < acc.shape[0] and theta_index > 0:
        if acc[rho_index + 1][theta_index - 1] >= acc[rho_index][theta_index]:
            return False
    if rho_index < acc.shape[0] and theta_index < acc.shape[1] - 1:
        if acc[rho_index + 1][theta_index + 1] >= acc[rho_index][theta_index]:
            return False
    # If none of the above cases is true, then it is a valid peak
    return True

"""
Finds the peak elements in the accumulator: these are lines in the image.
peak_thresh is the threshold used to determine what is a valid peak.
It should be the length in pixels of the smallest segment
we would consider a line.
"""
def findPeaks(acc, peak_thresh):
    peaks = []
    for rho in range(0, acc.shape[0]):
        for theta in range(0, acc.shape[1]):
            if is_peak(acc, rho, theta, peak_thresh):
                peaks.append([rho, theta])
    return peaks

def convert_angle(theta_index, theta_buckets):
    return np.pi * theta_index / theta_buckets

def convert_rho(rho_index, max_rho, rho_buckets):
    return (rho_index * 2.0 * max_rho / rho_buckets) - max_rho

"""
Note: angle_thresh is the number of theta indices, not a true angle
"""
def findPeakPairs(peaks, acc, angle_thresh, pixel_thresh, rho_thresh, max_rho, rho_buckets, theta_buckets):
    peakPairs = []
    for i in range(0, len(peaks)):
        for j in range(i+1, len(peaks)):
            cur1 = acc[peaks[i][0]][peaks[i][1]]
            cur2 = acc[peaks[j][0]][peaks[j][1]]
            
            if abs(peaks[i][1]-peaks[j][1]) < angle_thresh:
                if abs(cur1-cur2) < (pixel_thresh * (cur1 + cur2)/2):
                    rho_i = convert_rho(peaks[i][0], max_rho, rho_buckets)
                    theta_i = convert_angle(peaks[i][1], theta_buckets)
                    
                    rho_j = convert_rho(peaks[j][0], max_rho, rho_buckets)
                    theta_j = convert_angle(peaks[j][1], theta_buckets)
                    
                    if abs(rho_i - rho_j) > rho_thresh * (cur1 + cur2) / 2:
                        peakPairs.append([[rho_i, theta_i, cur1],[rho_j, theta_j, cur2]])
                    
    return peakPairs 
    #y coordinate is close to each other, and value in acc is close
    #return a list of pairs of lists (rho, theta, height2)

"""
Note: this assumes 'true values'
"""
def findParallelograms(peak_pairs, acc, pixel_thresh, parallel_angle_thresh):
    # peakPairs input format:
    # [pair1: [rho, theta, height], pair2: ...]
    
    # print("peakPairs:\n{}".format(peak_pairs))
    
    # find average rho, theta for the pairs:
    pair_averages = [];
    for pair in peak_pairs:
        theta_p = (pair[0][1] + pair[1][1]) / 2.0
        c_p = (pair[0][2] + pair[1][2]) / 2.0
        pair_averages.append([theta_p, c_p])

    parallelograms = []
    # for each pair of pairs:
    for i in range(0, len(peak_pairs)):
        for j in range(i + 1, len(peak_pairs)):
            peak_k = peak_pairs[i]
            average_k = pair_averages[i]
            peak_l = peak_pairs[j]
            average_l = pair_averages[j]

            delta_rho_k = abs(peak_k[0][0] - peak_k[1][0])
            delta_rho_l = abs(peak_l[0][0] - peak_l[1][0])

            alpha = abs(average_k[0] - average_l[0])

            d_1 = abs((delta_rho_k - average_l[1] * np.sin(alpha)) / delta_rho_k)
            d_2 = abs((delta_rho_l - average_k[1] * np.sin(alpha)) / delta_rho_l)

            if max(d_1, d_2) < pixel_thresh and alpha > parallel_angle_thresh:
                parallelograms.append([peak_k, peak_l])

    return parallelograms

def find_actual_perimeter(edge_image, parallelogram, max_rho, rho_buckets):
    # Loop along each pixel of each edge of the parallelogram,
    # as defined by our accumulator method.
    # If the pixel is on, add 1 to the perimeter.
    # If the accumulator parallelogram is real in the image, it should match up
    # with the lit pixels almost exactly and the actual perimeter returned
    # by this function should match the length of the accumulator lines almost
    # exactly. 

    vert0, vert1, vert2, vert3 = find_parallelogram_vertices(parallelogram, max_rho, rho_buckets, theta_buckets)

    perimeter = 0
    # For each segment on the parallelogram:
    # This is a pair of sequential vertices:
    # 0, 1; 1, 2; 2, 3; 3, 0

    # Loop the y coordinate in the range of y values included:
    # for instance, `for y in range(vert0[1], vert1[1])`
    # Use the formula for x from get_x in accum.py: 
    #   (rho / np.cos(theta)) - (y * np.tan(theta))
    # This gives us the x, y coordinate to check.
    # Perform the check at this pixel in the image
    # If the pixel is on, add 1 to perimeter
    
    return perimeter

"""
Requires: edge image
"""
def validate_parallelogram(edge_image, parallelogram, max_rho, rho_buckets, parallelogram_thresh):
    #TODO: get this data from parallelogram
    a = delta_rho_k / np.sin(alpha)
    b = delta_rho_l / np.sin(alpha)
    
    perim_estimate = 2 * (a + b)
    perim_actual = find_actual_perimeter(edge_image, parallelogram, max_rho, rho_buckets)
    
    if abs(perim_actual - perim_estimate) < perim_estimate * parallelogram_thresh:
        # Valid
        return True

"""
Finds, in Cartesian coordinates, the intersection of 2 lines defined in
the rho-theta parameterization.

If there is no intersection, it returns the point [-1, -1].
You have to check for this whenever you use this function.
"""
def find_intersection(rho_1, theta_1, rho_2, theta_2):
    print("rho1 {} theta1 {} rho2 {} theta2 {}".format(rho_1, theta_1, rho_2, theta_2))
    b = np.array([rho_1, rho_2])
    transform = np.array([[np.cos(theta_1), np.sin(theta_1)],
                          [np.cos(theta_2), np.sin(theta_2)]])
                          
    if np.linalg.det(transform) == 0:
        print("HELLO THIS IS AN ERROR YOU HAVE A SINGULAR MATRIX")
        return [-1.0, -1.0]
    # print("transform \n{}".format(transform))
    intersection = np.dot(np.linalg.inv(transform), b)
    # print("intersection \n{}".format(intersection))
    cartesian = intersection.reshape((2))
    return cartesian

"""
Vertex 1: k0, l0
Vertex 2: k0, l1
Vertex 2: k1, l1
Vertex 3: k1, l0

Vertices are chosen in this order so that they go sequentially around the 
parallelogram. We don't know whether it's CW or CCW, or what the orientation 
of each vertex is with regard to the others.
"""
def find_parallelogram_vertices(parallelogram, max_rho, rho_buckets, theta_buckets):
    print("parallelogram \n{}".format(parallelogram))
    peak_k = parallelogram[0]
    peak_l = parallelogram[1]
    
    rho_k_0 = peak_k[0][0]
    theta_k_0 = peak_k[0][1]
    rho_k_1 = peak_k[1][0]
    theta_k_1 = peak_k[1][1]
    rho_l_0 = peak_l[0][0]
    theta_l_0 = peak_l[0][1]
    rho_l_1 = peak_l[1][0]
    theta_l_1 = peak_l[1][1]
    
    intersection_1 = find_intersection(rho_k_0, theta_k_0, rho_l_0, theta_l_0)
    print("Intersection 1: x {}    y {}".format(intersection_1[0], intersection_1[1]))
    intersection_2 = find_intersection(rho_k_0, theta_k_0, rho_l_1, theta_l_1)
    print("Intersection 2: x {}    y {}".format(intersection_2[0], intersection_2[1]))
    intersection_3 = find_intersection(rho_k_1, theta_k_1, rho_l_0, theta_l_0)
    print("Intersection 3: x {}    y {}".format(intersection_3[0], intersection_3[1]))
    intersection_4 = find_intersection(rho_k_1, theta_k_1, rho_l_1, theta_l_1)
    print("Intersection 4: x {}    y {}".format(intersection_4[0], intersection_4[1]))

    return intersection_1, intersection_2, intersection_3, intersection_4
#tiling
