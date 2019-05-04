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
Point = collections.namedtuple("Point", ["x", "y"])

class Edge:
    """
    An Edge is a segment defined mostly by a single Peak line.
    It is bordered (constrained) by two bordering lines. These lines come 
    from accumulator Peaks.
    
    Fields:
    endpoints (x, y)
    endpoint_[top bottom]: the endpoints of the edge segment.
                           They are sorted by their vertical (y) axis position.
    peak parameterization:
        rho (radius from origin, pixels)
        theta (angle from vertical, degrees)

    base_peak: the Peak object that defines the edge segment
    """

    """
    Input:
    edge_peak: the Peak object representing the line that contains the edge
    border_peak_[1 2]: the bordering, or constraining, peaks that define the
                       edge's endpoints
    """
    def __init__(self, edge_peak, border_peak_1, border_peak_2):
        self.base_peak = edge_peak
        
        self.rho = edge_peak.rho
        self.theta = edge_peak.theta
        
        endpoint_1 = find_intersection(self.rho, self.theta, border_peak_1.rho, border_peak_1.theta)
        endpoint_2 = find_intersection(self.rho, self.theta, border_peak_2.rho, border_peak_2.theta)

        if endpoint_1[1] > endpoint_2[1]:
            self.endpoint_top = endpoint_1
            self.endpoint_bottom = endpoint_2
        else:
            self.endpoint_top = endpoint_2
            self.endpoint_bottom = endpoint_1

    """
    Determines if the entire edge is contained in the image.

    Input:
    image: the numpy array for the image.
           This is used to get the boundaries of the image.
    """
    def in_image(self, image):
        y_size = image.shape[0]
        x_size = image.shape[1]

        if self.endpoint_top[1] < 0 or self.endpoint_top[1] >= y_size:
            return False
        if self.endpoint_bottom[1] < 0 or self.endpoint_bottom[1] >= y_size:
            return False
        if self.endpoint_top[0] < 0 or self.endpoint_top[0] >= x_size:
            return False
        if self.endpoint_bottom[0] < 0 or self.endpoint_bottom[0] >= x_size:
            return False

        return True

    def get_x(self, y):
        return (self.rho / np.cos(self.theta)) - (y * np.tan(self.theta))

class Parallelogram:
    """
    Fields:
    edges: list of length 4
    alpha: the angle between the peak pairs

    It also contains the original PeakPairs that define the parallelogram:
    peak pairs: pair_k
                pair_l
    """

    """
    Input:
    The PeakPair objects that make up the parallelogram.
    The order of which is called k and which is l does not matter.
    We just have to define them arbitrarily.
    """
    def __init__(self, pair_k, pair_l):
        self.pair_k = pair_k
        self.pair_l = pair_l
        self.edges = find_parallelogram_edges(self.pair_k, self.pair_l)

        self.alpha = abs(self.pair_k.average_theta - self.pair_l.average_theta)

    def old_list_format(self):
        return (self.pair_k, self.pair_l)

class PeakPair:
    """
    Fields:
    Initial information. These are the values that make up the accumulator 
    peaks in the pair.
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
    distance between peaks: peak_distance

    It also includes individual edge angles, since we need to use that 
    sometimes. However, for the most part, the averages are used because
    that corresponds to a true parallelogram.
    """
    def __init__(self, peak_i, peak_j):
        self.peak_i = peak_i
        self.peak_j = peak_j

        self.rho_i = peak_i.rho
        self.rho_j = peak_j.rho
        self.theta_i = peak_i.theta
        self.theta_j = peak_j.theta
        self.average_theta = (peak_i.theta + peak_j.theta) / 2.0
        self.average_height = (peak_i.height + peak_j.height) / 2.0

        self.peak_distance = abs(self.rho_i - self.rho_j)
    
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
def findPeakPairs(peaks, acc, angle_thresh, pixel_thresh, rho_thresh, max_rho, rho_buckets):
    peak_pairs = []
    for i in range(0, len(peaks)):
        for j in range(i+1, len(peaks)):
            height_i = peaks[i].height
            height_j = peaks[j].height
            
            if abs(peaks[i].theta_bucket - peaks[j].theta_bucket) < angle_thresh:
                if abs(height_i - height_j) < (pixel_thresh * (height_i + height_j)/2):
                    rho_i = convert_rho(peaks[i].rho_bucket, max_rho, rho_buckets)
                    
                    rho_j = convert_rho(peaks[j].rho_bucket, max_rho, rho_buckets)
                    
                    if abs(rho_i - rho_j) > rho_thresh * (height_i + height_j) / 2:
                        peak_pairs.append(PeakPair(peaks[i], peaks[j]))
                    
    return peak_pairs
    #y coordinate is close to each other, and value in acc is close
    #return a list of pairs of lists (rho, theta, height2)

"""
Note: this assumes 'true values'
"""
def findParallelograms(peak_pairs, acc, pixel_thresh, parallel_angle_thresh):
    parallelogram_objects = []
    parallelograms = []
    # for each pair of pairs:
    for i in range(0, len(peak_pairs)):
        for j in range(i + 1, len(peak_pairs)):
            pair_k = peak_pairs[i]
            pair_l = peak_pairs[j]

            alpha = abs(pair_k.average_theta - pair_l.average_theta)

            d_1a = abs((pair_k.peak_distance - pair_l.average_height * np.sin(alpha)) / pair_k.peak_distance)
            d_2a = abs((pair_l.peak_distance - pair_k.average_height * np.sin(alpha)) / pair_l.peak_distance)

            if max(d_1a, d_2a) < pixel_thresh and alpha > parallel_angle_thresh:
                parallelogram_objects.append((pair_k, pair_l))
                parallelograms.append(Parallelogram(pair_k, pair_l))
                assert(parallelogram_objects[-1] == parallelograms[-1].old_list_format())

    return parallelogram_objects

def find_actual_perimeter(parallelogram, edge_image):
    # Loop along each pixel of each edge of the parallelogram,
    # as defined by our accumulator method.
    # If the pixel is on, add 1 to the perimeter.
    # If the accumulator parallelogram is real in the image, it should match up
    # with the lit pixels almost exactly and the actual perimeter returned
    # by this function should match the length of the accumulator lines almost
    # exactly. 
    pair_k = parallelogram[0]
    pair_l = parallelogram[1]
    edges = find_parallelogram_edges(pair_k, pair_l)

    perimeter = 0
    # For each segment on the parallelogram:
    # This is a pair of sequential vertices:
    # 0, 1; 1, 2; 2, 3; 3, 0
    for edge in edges:
        if not edge.in_image(edge_image):
            return False

        y_bottom = int(edge.endpoint_bottom[1])
        y_top = int(edge.endpoint_top[1])

        for y in range(y_bottom, y_top + 1):
            x = int(edge.get_x(y))
            
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
def validate_parallelogram(parallelogram, edge_image, parallelogram_thresh):
    #TODO: get this data from parallelogram
    pair_k = parallelogram[0]
    pair_l = parallelogram[1]

    # Calculate estimated perimeter of parallelogram based on the "lines" 
    # (peaks) that make it up.
    # This uses the formula p = 2(a + b), 
    # where a and b are the lengths of the sides.
    alpha = abs(pair_k.average_theta - pair_l.average_theta)
    side_a = pair_k.peak_distance / np.sin(alpha)
    side_b = pair_l.peak_distance / np.sin(alpha)
    perim_estimate = 2 * (side_a + side_b)

    perim_actual = find_actual_perimeter(parallelogram, edge_image)
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
def find_parallelogram_vertices(parallelogram):
    pair_k = parallelogram[0]
    pair_l = parallelogram[1]

    rho_k_i = pair_k.rho_i
    theta_k_i = pair_k.theta_i
    rho_k_j = pair_k.rho_j
    theta_k_j = pair_k.theta_j
    rho_l_i = pair_l.rho_i
    theta_l_i = pair_l.theta_i
    rho_l_j = pair_l.rho_j
    theta_l_j = pair_l.theta_j

    intersection_0 = find_intersection(rho_k_i, theta_k_i, rho_l_i, theta_l_i)
    intersection_1 = find_intersection(rho_k_i, theta_k_i, rho_l_j, theta_l_j)
    intersection_2 = find_intersection(rho_k_j, theta_k_j, rho_l_j, theta_l_j)
    intersection_3 = find_intersection(rho_k_j, theta_k_j, rho_l_i, theta_l_i)

    return intersection_0, intersection_1, intersection_2, intersection_3

def find_parallelogram_edges(pair_k, pair_l):
    side_0 = Edge(pair_k.peak_i, pair_l.peak_i, pair_l.peak_j)
    side_1 = Edge(pair_l.peak_i, pair_k.peak_i, pair_k.peak_j)
    side_2 = Edge(pair_k.peak_j, pair_l.peak_i, pair_l.peak_j)
    side_3 = Edge(pair_l.peak_j, pair_k.peak_i, pair_k.peak_j)

    return side_0, side_1, side_2, side_3
