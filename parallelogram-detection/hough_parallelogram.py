import numpy as np
import cv2
import collections

"""
Using terminology from paper:
Rho is the radius from the origin in pixels.
Theta is the angle from the vertical in radians.
The buckets are the accumulator indices corresponding to each, while rho and theta are true radius
and angle values.
The height is the height of the accumulator array at the point (rho, theta). In the paper, it is
denoted C.
"""
Peak = collections.namedtuple("Peak", ["rho_bucket", "rho", "theta_bucket", "theta", "height"])
# Note for refactor: peak[0] = rho, peak[1] = theta

class Edge:
    """
    Fields:
    endpoints (x, y)
    peak parameterization:
        rho (radius from origin, pixels)
        theta (angle from vertical, degrees)
    """
    def __init__(self, actual, left, right):
        self.endpoints = []
        self.rho = actual.rho
        self.theta = actual.theta
        
        self.endpoints.append(find_intersection(self.rho, self.theta, left.rho, left.theta))
        self.endpoints.append(find_intersection(self.rho, self.theta, right.rho, right.theta))

class Parallelogram:
    """
    Fields:
    edges: list of length 4
    peak pairs: pair_k
                pair_l
    """
    def __init__(self, peak_pairs, max_rho, rho_buckets, theta_buckets):
        self.pair_k = peak_pairs[0]
        self.pair_l = peak_pairs[1]
        self.edges = find_parallelogram_edges(peak_pairs, max_rho, rho_buckets, theta_buckets)

class PeakPair:
    """
    Fields:
    Initial information. These are the values that make up the accumulator peaks in the pair.
    peaks:  peak_i
            peak_j
    
    Pair information. These values are calculated from the peak and are the 
    properties of the pair.
    A pair is a set of parallel and equally long segments. It is defined by 
    an angle to the vertical, average_theta; a length, average_height; and 
    the radii of the segments themselves, rho_i and rho_j.
    line radii: rho_i
                rho_j
    average values: average_theta
                    average_height
    """
    def __init__(self, peak_i, peak_j):
        self.peak_i = peak_i
        self.peak_j = peak_j

        self.rho_i = peak_i.rho
        self.rho_j = peak_j.rho
        self.average_theta = (peak_i.theta + peak_j.theta) / 2.0
        self.average_height = (peak_i.height + peak_j.height) / 2.0
    
    def old_list_format(self):
        return (self.peak_i, self.peak_j)

def convert_angle_radians(theta_index, theta_buckets):
    return np.pi * theta_index / theta_buckets

def convert_rho(rho_index, max_rho, rho_buckets):
    return (rho_index * 2.0 * max_rho / rho_buckets) - max_rho

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
def is_peak(acc, rho_bucket, theta_bucket, peak_thresh):
    if acc[rho_bucket, theta_bucket] < peak_thresh:
        return False

    # Check that acc[rho_index, theta_index] is greater than all neighboring values,
    # making sure that we don't go outside array bounds
    # Check vertical
    if rho_bucket > 0:
        if acc[rho_bucket - 1][theta_bucket] >= acc[rho_bucket][theta_bucket]:
            return False
    if rho_bucket < acc.shape[0] - 1:
        if acc[rho_bucket + 1][theta_bucket] >= acc[rho_bucket][theta_bucket]:
            return False
    # Check horizontal
    if theta_bucket > 0:
        if acc[rho_bucket][theta_bucket - 1] >= acc[rho_bucket][theta_bucket]:
            return False
    if theta_bucket < acc.shape[1] - 1:
        if acc[rho_bucket][theta_bucket + 1] >= acc[rho_bucket][theta_bucket]:
            return False
    # Check diagonal above
    if rho_bucket > 0 and theta_bucket > 0:
        if acc[rho_bucket - 1][theta_bucket - 1] >= acc[rho_bucket][theta_bucket]:
            return False
    if rho_bucket > 0 and theta_bucket < acc.shape[1] - 1:
        if acc[rho_bucket - 1][theta_bucket + 1] >= acc[rho_bucket][theta_bucket]:
            return False
    # Check diagonal below
    if rho_bucket < acc.shape[0] and theta_bucket > 0:
        if acc[rho_bucket + 1][theta_bucket - 1] >= acc[rho_bucket][theta_bucket]:
            return False
    if rho_bucket < acc.shape[0] and theta_bucket < acc.shape[1] - 1:
        if acc[rho_bucket + 1][theta_bucket + 1] >= acc[rho_bucket][theta_bucket]:
            return False
    # If none of the above cases is true, then it is a valid peak
    return True

"""
Finds the peak elements in the accumulator: these are lines in the image.
peak_thresh is the threshold used to determine what is a valid peak.
It should be the length in pixels of the smallest segment
we would consider a line.
"""
def findPeaks(acc, enhanced_acc, peak_thresh, max_rho, rho_buckets, theta_buckets):
    peaks = []
    for rho_bucket in range(0, enhanced_acc.shape[0]):
        for theta_bucket in range(0, enhanced_acc.shape[1]):
            if is_peak(enhanced_acc, rho_bucket, theta_bucket, peak_thresh):
                rho = convert_rho(rho_bucket, max_rho, rho_buckets)
                theta = convert_angle_radians(theta_bucket, theta_buckets)
                height = acc[rho_bucket][theta_bucket]
                peak = Peak(rho_bucket=rho_bucket, rho=rho, theta_bucket=theta_bucket, theta=theta, height=height)
                peaks.append(peak)
                
    return peaks

"""
Note: angle_thresh is the number of theta indices, not a true angle
"""
def findPeakPairs(peaks, acc, angle_thresh, pixel_thresh, rho_thresh, max_rho, rho_buckets, theta_buckets):
    peak_pairs = []
    peak_pairs_obj = []
    for i in range(0, len(peaks)):
        for j in range(i+1, len(peaks)):
            cur1 = acc[peaks[i].rho_bucket][peaks[i].theta_bucket]
            cur2 = acc[peaks[j].rho_bucket][peaks[j].theta_bucket]
            height_i = peaks[i].height
            height_j = peaks[j].height
            assert(height_i == cur1)
            assert(height_j == cur2)
            
            if abs(peaks[i].theta_bucket - peaks[j].theta_bucket) < angle_thresh:
                if abs(cur1 - cur2) < (pixel_thresh * (cur1 + cur2)/2):
                    rho_i = convert_rho(peaks[i].rho_bucket, max_rho, rho_buckets)
                    theta_i = convert_angle_radians(peaks[i].theta_bucket, theta_buckets)
                    
                    rho_j = convert_rho(peaks[j].rho_bucket, max_rho, rho_buckets)
                    theta_j = convert_angle_radians(peaks[j].theta_bucket, theta_buckets)
                    
                    if abs(rho_i - rho_j) > rho_thresh * (cur1 + cur2) / 2:
                        # peak_pairs.append(([rho_i, theta_i, cur1],[rho_j, theta_j, cur2]))
                        peak_pairs.append((peaks[i], peaks[j]))
                        new_pair = PeakPair(peaks[i], peaks[j])
                        peak_pairs_obj.append(new_pair)
                        assert((peaks[i], peaks[j]) == new_pair.old_list_format())
                    
    return peak_pairs_obj
    #y coordinate is close to each other, and value in acc is close
    #return a list of pairs of lists (rho, theta, height2)

"""
Note: this assumes 'true values'
"""
def findParallelograms(peak_pairs_obj, acc, pixel_thresh, parallel_angle_thresh, max_rho, rho_buckets, theta_buckets):
    # peakPairs input format:
    # [pair1: [rho, theta, height], pair2: ...]
    peak_pairs = []
    for obj in peak_pairs_obj:
        peak_pairs.append(obj.old_list_format())
    
    # print("peakPairs:\n{}".format(peak_pairs))
    
    # find average rho, theta for the pairs:
    pair_averages = []
    for pair in peak_pairs:
        theta_p = (pair[0].theta + pair[1].theta) / 2.0
        c_p = (pair[0].height + pair[1].height) / 2.0
        pair_averages.append([theta_p, c_p])

    parallelograms = []
    parallelogram_objects = []
    # for each pair of pairs:
    for i in range(0, len(peak_pairs)):
        for j in range(i + 1, len(peak_pairs)):
            pair_k = peak_pairs[i]
            average_k = pair_averages[i]
            pair_l = peak_pairs[j]
            average_l = pair_averages[j]

            delta_rho_k = abs(pair_k[0].rho - pair_k[1].rho)
            delta_rho_l = abs(pair_l[0].rho - pair_l[1].rho)

            alpha = abs(average_k[0] - average_l[0])

            d_1 = abs((delta_rho_k - average_l[1] * np.sin(alpha)) / delta_rho_k)
            d_2 = abs((delta_rho_l - average_k[1] * np.sin(alpha)) / delta_rho_l)

            if max(d_1, d_2) < pixel_thresh and alpha > parallel_angle_thresh:
                parallelogram_pairs = [pair_k, pair_l]
                parallelograms.append(parallelogram_pairs)
                obj = Parallelogram(parallelogram_pairs, max_rho, rho_buckets, theta_buckets)
                parallelogram_objects.append(obj)

    return parallelograms

def find_actual_perimeter(edge_image, parallelogram, max_rho, rho_buckets, theta_buckets):
    test_image = np.zeros((edge_image.shape[0], edge_image.shape[1], 3))
    test_overlay = cv2.cvtColor(edge_image, cv2.COLOR_GRAY2BGR)
    colors = ((255, 255, 255), (255, 0, 0), (0, 0, 255), (0, 255, 0))

    # Loop along each pixel of each edge of the parallelogram,
    # as defined by our accumulator method.
    # If the pixel is on, add 1 to the perimeter.
    # If the accumulator parallelogram is real in the image, it should match up
    # with the lit pixels almost exactly and the actual perimeter returned
    # by this function should match the length of the accumulator lines almost
    # exactly. 
    
    pair_k = parallelogram[0]
    pair_l = parallelogram[1]
    
    edges = [pair_k[0], pair_l[1], pair_k[1], pair_l[0]]

    # Returned in format: [x, y]
    vertices = find_parallelogram_vertices(parallelogram, max_rho, rho_buckets, theta_buckets)

    perimeter = 0
    # For each segment on the parallelogram:
    # This is a pair of sequential vertices:
    # 0, 1; 1, 2; 2, 3; 3, 0
    for i in range(0, 4):
        # Get the index of the second vertex: wraps back to 0 after 3
        j = (i + 1) % 4;
        vert0 = vertices[i]
        vert1 = vertices[j]
        rho = edges[i].rho
        theta = edges[i].theta

        # Check that all vertices are actually in the image.
        # If they're not, this can't be a valid parallelogram in the image
        if vert0[0] < 0 or vert0[0] >= edge_image.shape[1]:
            return False
        if vert0[1] < 0 or vert0[1] >= edge_image.shape[0]:
            return False
        if vert1[0] < 0 or vert1[0] >= edge_image.shape[1]:
            return False
        if vert1[1] < 0 or vert1[1] >= edge_image.shape[0]:
            return False

        draw0 = (int(vert0[0]), int(vert0[1]))
        draw1 = (int(vert1[0]), int(vert1[1]))
        # print("draw0 {} draw1 {}".format(draw0, draw1))
        # cv2.line(test_overlay, draw0, draw1, colors[i])
        # cv2.imshow("Testing perimeter", test_overlay)

        y0 = int(vert0[1])
        y1 = int(vert1[1])
        # If y1 > y0, switch them so the range still works as expected
        if y0 > y1:
            tmp = y0
            y0 = y1
            y1 = tmp
        for y in range(y0, y1 + 1):
            x = int((rho / np.cos(theta)) - (y * np.tan(theta)))
            # print("x {} y {}".format(x, y))
            # If this pixel is on, add 1 to perimeter
            if edge_image[y][x] != 0:
            #     print("Hey! Perimeter! {}".format(perimeter))
                perimeter = perimeter + 1
        
        # Loop the y coordinate in the range between the included values
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
def validate_parallelogram(edge_image, parallelogram, max_rho, rho_buckets, theta_buckets, parallelogram_thresh):
    #TODO: get this data from parallelogram
    pair_k = parallelogram[0]
    pair_l = parallelogram[1]
    delta_rho_k = abs(pair_k[0].rho - pair_k[1].rho)
    delta_rho_l = abs(pair_l[0].rho - pair_l[1].rho)
    average_alpha_k = (pair_k[0].theta + pair_k[1].theta) / 2.0
    average_alpha_l = (pair_l[0].theta + pair_l[1].theta) / 2.0
    alpha = abs(average_alpha_k - average_alpha_l)
    a = delta_rho_k / np.sin(alpha)
    b = delta_rho_l / np.sin(alpha)
    
    perim_estimate = 2 * (a + b)
    perim_actual = find_actual_perimeter(edge_image, parallelogram, max_rho, rho_buckets, theta_buckets)
    if perim_actual == False:
        return False, 1
    
    if abs(perim_actual - perim_estimate) < perim_estimate * parallelogram_thresh:
        # Valid
        # print("                  Valid: estimate {} actual {} ======================".format(perim_estimate, perim_actual))
        return True, abs(perim_actual - perim_estimate) / perim_estimate
    else:
        # print("INVALID: estimate {} actual {}".format(perim_estimate, perim_actual))
        return False, abs(perim_actual - perim_estimate) / perim_estimate

"""
Finds, in Cartesian coordinates, the intersection of 2 lines defined in
the rho-theta parameterization.

If there is no intersection, it returns the point [-1, -1].
You have to check for this whenever you use this function.
"""
def find_intersection(rho_1, theta_1, rho_2, theta_2):
    # print("rho1 {} theta1 {} rho2 {} theta2 {}".format(rho_1, theta_1, rho_2, theta_2))
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

Resulting format: list of 4 vertex lists
Each vertex is a list of [x, y] coordinates.
A [-1, -1] vertex is taken to be invalid.
"""
def find_parallelogram_vertices(parallelogram, max_rho, rho_buckets, theta_buckets):
    pair_k = parallelogram[0]
    pair_l = parallelogram[1]
    
    rho_k_0 = pair_k[0].rho
    theta_k_0 = pair_k[0].theta
    rho_k_1 = pair_k[1].rho
    theta_k_1 = pair_k[1].theta
    rho_l_0 = pair_l[0].rho
    theta_l_0 = pair_l[0].theta
    rho_l_1 = pair_l[1].rho
    theta_l_1 = pair_l[1].theta
    
    intersection_0 = find_intersection(rho_k_0, theta_k_0, rho_l_0, theta_l_0)
    # print("Intersection 0: x {}    y {}".format(intersection_1[0], intersection_1[1]))
    intersection_1 = find_intersection(rho_k_0, theta_k_0, rho_l_1, theta_l_1)
    # print("Intersection 1: x {}    y {}".format(intersection_2[0], intersection_2[1]))
    intersection_2 = find_intersection(rho_k_1, theta_k_1, rho_l_1, theta_l_1)
    # print("Intersection 2: x {}    y {}".format(intersection_4[0], intersection_4[1]))
    intersection_3 = find_intersection(rho_k_1, theta_k_1, rho_l_0, theta_l_0)
    # print("Intersection 3: x {}    y {}".format(intersection_3[0], intersection_3[1]))

    return intersection_0, intersection_1, intersection_2, intersection_3

def find_parallelogram_edges(parallelogram, max_rho, rho_buckets, theta_buckets):
    # print("parallelogram \n{}".format(parallelogram))
    peak_k = parallelogram[0]
    peak_l = parallelogram[1]

    side_0 = Edge(peak_k[0], peak_l[0], peak_l[1])
    side_1 = Edge(peak_l[0], peak_k[0], peak_k[1])
    side_2 = Edge(peak_k[1], peak_l[1], peak_l[0])
    side_3 = Edge(peak_l[1], peak_k[1], peak_k[0])

    return side_0, side_1, side_2, side_3
