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

def findPeakPairs():
    return

def findParallelograms(peakPairs):
    return




#tiling
