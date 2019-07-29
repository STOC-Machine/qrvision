from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)

HEIGHT = 1200
WIDTH = 1200
area_threshold = 3

def distance(point1, point2):
    return (point1[0] - point2[0])**2 + (point1[1] - point2[1])**2

def thresh_callback(val, src, position):
    # Convert image to gray and blur it
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    src_gray = cv.blur(src_gray, (3,3))

    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # hull_list = []
    # for i in range(len(contours)):
    #     hull = cv.convexHull(contours[i])
    #     hull_list.append(hull)

    # # Draw contours + hull results
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # for i in range(len(contours)):
    #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    #     # cv.drawContours(drawing, contours, i, color)
    #     cv.drawContours(drawing, hull_list, i, color)
    # # Show in a window
    # cv.imshow('Hulls', drawing)

    # # Draw hulls result
    # # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # drawing = src
    # for i in range(len(contours)):
    #     color = (256,0,0)
    #     cv.drawContours(drawing, contours, i, color)
    #     # cv.drawContours(drawing, hull_list, i, color)
    # # Show in a window
    # cv.imshow('Contours', drawing)

    # Find the convex hull object for all contours
    hull_list = []

    # Combine all contours into one
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    drawing = src
    for i in range( len(contours)):
        if cv.contourArea(contours[i]) > area_threshold:
            contours[0] = contours[i]
            break
    for i in range(1,len(contours)):
        if cv.contourArea(contours[i]) > area_threshold:
            contours[0] = np.vstack((contours[0], contours[i]))
        
    hull = cv.convexHull(contours[0])
    hull_list.append(hull)
    cv.drawContours(drawing, hull_list, 0, (256,125,10))
    hull_list.append(hull)

    cv.imshow('Hull', drawing)

    width_src, height_src, channels_src = src.shape

    # top = hull[0][0][1]
    # down = hull[0][0][1]
    # left = hull[0][0][0]
    # right = hull[0][0][0]
    # for point in hull:
    #     top = max(top, point[0][1])
    #     down = min(down, point[0][1])
    #     left = min(left, point[0][0])
    #     right = max(right, point[0][0])

    # Extract corner of the hull
    corner_top_left = 0
    for i in range(1, len(hull) ):
        if distance(hull[i][0], [0, 0]) < distance(hull[corner_top_left][0], [0, 0]):
            corner_top_left = i
    corner_top_right = 0
    for i in range(1, len(hull) ):
        if distance(hull[i][0], [width_src, 0]) < distance(hull[corner_top_right][0], [width_src, 0]):
            corner_top_right = i
    corner_bottom_left = 0
    for i in range(1, len(hull) ):
        if distance(hull[i][0], [0, height_src]) < distance(hull[corner_bottom_left][0], [0, height_src]):
            corner_bottom_left = i
    corner_bottom_right = 0
    for i in range(1, len(hull) ):
        if distance(hull[i][0], [width_src, height_src]) < distance(hull[corner_bottom_right][0], [width_src, height_src]):
            corner_bottom_right = i

    if position == "bottom right":
        # Find the point that will map into the middle of result
        pts1 = np.float32([hull[corner_top_left][0], hull[corner_bottom_left][0], hull[corner_top_right][0]])
        pts2 = np.float32([ [600,600], [600,1200], [1200,600] ])
        M = cv.getAffineTransform(pts1,pts2)

        dst = cv.warpAffine(src, M, (height_src,width_src))
        # dst = cv.warpPerspective(src,M,(300,300))
        result = cv.imread("qr/result.jpg")
        result[600:1200, 600:1200] = dst[600:1200, 600:1200]
        cv.imwrite("qr/result.jpg", result)

    elif position == "bottom left":
        # Find the point that will map into the middle of result
        pts1 = np.float32([hull[corner_top_right][0], hull[corner_bottom_right][0], hull[corner_top_left][0]])
        pts2 = np.float32([ [600,600], [600,1200], [0,600] ])

        M = cv.getAffineTransform(pts1,pts2)

        dst = cv.warpAffine(src, M, (height_src,width_src))
        # dst = cv.warpPerspective(src,M,(300,300))

        result = cv.imread("qr/result.jpg")
        # result = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        result[600:1200, 0:600] = dst[600:1200, 0:600]
        cv.imwrite("qr/result.jpg", result)


    elif position == "top right":
        # Find the point that will map into the middle of result
        pts1 = np.float32([hull[corner_bottom_left][0], hull[corner_bottom_right][0], hull[corner_top_left][0]])
        pts2 = np.float32([ [600,600], [1200,600], [600,0] ])

        M = cv.getAffineTransform(pts1,pts2)
        dst = cv.warpAffine(src, M, (height_src,width_src))

        result = cv.imread("qr/result.jpg")
        result[0:600, 600:1200] = dst[0:600, 600:1200]
        cv.imwrite("qr/result.jpg", result)

    else: # top left
        pts1 = np.float32([hull[corner_bottom_right][0], hull[corner_top_right][0], hull[corner_bottom_left][0]])
        pts2 = np.float32([ [600,600], [600,0], [0,600] ])

        M = cv.getAffineTransform(pts1,pts2)
        dst = cv.warpAffine(src, M, (height_src,width_src))

        result = cv.imread("qr/result.jpg")
        result[0:600, 0:600] = dst[0:600, 0:600]
        cv.imwrite("qr/result.jpg", result)


def decode_QR(image):
    from pyzbar.pyzbar import decode
    from PIL import Image
    # print( decode(Image.open(image)))
    print( decode(Image.open(image))[0].data)

def process_image(image):
    # Load source image
    src = cv.imread(cv.samples.findFile(image))
    src = cv.resize(src, (int(WIDTH),int(HEIGHT) ) )
    cv.imwrite("qr/resize.jpg", src)
    if src is None:
        print('Could not open or find the image:', image)
        exit(0)

    # decode_QR(image)
    # # Create Window
    # source_window = 'Source'
    # cv.namedWindow(source_window)
    # cv.imshow(source_window, src)
    max_thresh = 255
    # thresh = 100 # initial threshold
    thresh = 130 # initial threshold
    # cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
    thresh_callback(thresh, src, "bottom left" )
    # 1: bottom right, 2: bottom left, 3: top left
    # cv.waitKey()


parser = argparse.ArgumentParser(description='Code for Convex Hull tutorial.')
parser.add_argument('--input', help='Path to input image.', default='qr/qr1.png')
args = parser.parse_args()
process_image(args.input)
# decode_QR(args.input)
cv.waitKey()


