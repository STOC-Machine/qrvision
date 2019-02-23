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

def findPeakPairs(peaks, acc, angle_thresh, pixel_thresh):
    peakPairs = []
    for i in range(0, len(peaks)):
        for j in range(i+1, len(peaks)):
            cur1 = acc[peaks[i][0]][peaks[i][1]]
            cur2 = acc[peaks[j][0]][peaks[j][1]]
            if abs(peaks[i][1]-peaks[j][1]) < angle_thresh:
                if abs(cur1-cur2) < (pixel_thresh * (cur1 + cur2)/2):
                    peakPairs.append([[peaks[i][0], peaks[i][1],cur1],[peaks[j][0],peaks[j][1],cur2]])
    return peakPairs
    #y coordinate is close to each other, and value in acc is close
    #return a list of pairs of lists (rho, theta, height2)

def findParallelograms(acc, peak_pairs, pixel_thresh):
    # peakPairs input format:
    # [pair1: [rho, theta, height], pair2: ...]

    pair_averages = [];
    # find average rho, theta for the pairs:
    for pair in peakPairs:
        theta_p = average(pair[0][1], pair[[1][1]])
        c_p = average(pair[0][2], pair[1][2])
        pair_averages.append([theta_p, c_p])

    parallelograms = []
    # for each pair of pairs:
    for i in range(0, len(peak_pairs)):
        for j in range(i + 1, len(peak_pairs)):
            peak_k = peak_pairs[i]
            average_k = pair_averages[i]
            peak_l = peak_pairs[j]
            average_l = pair_averages[j]

            # eta = abs(rho_k^2 - rho_l^2)
            delta_rho_k = abs(peak1[0][0]**2 - peak1[1][0]**2)
            delta_rho_l = abs(peak2[0][0]**2 - peak2[1][0]**2)

            alpha = average_k[0] - average_l[0]

            d_1 = (delta_rho_k - average_l[1] * np.sin(alpha)) / delta_rho_k
            d_2 = (delta_rho_l - average_k[1] * np.sin(alpha)) / delta_rho_l

            if max(d_1, d_2) < pixel_thresh:
                parallelograms.append([peak_k, peak_l])

    return parallelograms




#tiling
