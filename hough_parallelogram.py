import numpy as np

#section 1:find parallelograms in image
def accumulator():
    return
def enhance(acc, h, w):
    enhanced = np.copy(acc)
    
    # Outer loop: go by blocks
    minRho = minTheta = 0
    maxRho = h - 1
    maxTheta = w - 1
    while acc.shape[0] > maxRho:
        while acc.shape[1] > maxTheta:
            # get sum of region
            sum = 0
            for rho in range(minRho, maxRho):
                for theta in range(minTheta, maxTheta):
                    sum = sum + acc[rho][theta]
            
            if sum != 0:
                # Inner loop: element by element in the region and save to enhanced
                for rho in range(minRho, maxRho):
                    for theta in range(minTheta, maxTheta):
                        numerator = h * w * acc[rho][theta]
                        enhanced[rho][theta] = numerator / sum
            minTheta += w
            maxTheta += w
        minRho += h
        maxRho += h
    
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
