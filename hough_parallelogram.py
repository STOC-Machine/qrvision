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

def findPeakPairs():
    return

def findParallelograms(peakPairs):
    return




#tiling
