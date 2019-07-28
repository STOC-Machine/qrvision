# import cv2
# import numpy as np
# img = cv2.imread("test.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray, 75, 150)
# lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, maxLineGap=100000)
# for line in lines:
#    x1, y1, x2, y2 = line[0]
#    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 128), 1)
# cv2.imshow("linesEdges", edges)
# cv2.imshow("linesDetected", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import sys

# def extract_corners(image):
#     """
#     Find the 4 corners of a binary image
#     :param image: binary image
#     :return: 4 main vertices or None
#     """
#     cnts, _ = cv2.findContours(image.copy(),
#                                cv2.RETR_EXTERNAL,
#                                cv2.CHAIN_APPROX_SIMPLE)[-2:]
#     cnt = cnts[0]
#     _, _, h, w = cv2.boundingRect(cnt)
#     epsilon = min(h, w) * 0.5
#     vertices = cv2.approxPolyDP(cnt, epsilon, True)
#     vertices = cv2.convexHull(vertices, clockwise=True)
#     vertices = self.correct_vertices(vertices)

#     return vertices 
# extract_corners("qrcode.png")


from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
rng.seed(12345)

HEIGHT = 1200
WIDTH = 1200
def thresh_callback(val):
    threshold = val
    # Detect edges using Canny
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    # Find contours
    contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Find the convex hull object for each contour
    hull_list = []

    # print(contours[0][1])
    # print("Contour 0", contours[0])
    # print(type(contours[0]))
    for i in range(1,len(contours)):
        # for j in range(len(contours[i])):
        contours[0] = np.vstack((contours[0], contours[i]))
        # print("Contour",i, contours[i])
        # print("Contour 0", contours[0])
        # print()
        # break
    hull = cv.convexHull(contours[0])
    hull_list.append(hull)
    # print(hull)

    # width = src.size().width
    # height = src.size().height
    # print(width,height)

    top = hull[0][0][1]
    down = hull[0][0][1]
    left = hull[0][0][0]
    right = hull[0][0][0]

    for point in hull:
        top = max(top, point[0][1])
        down = min(down, point[0][1])
        left = min(left, point[0][0])
        right = max(right, point[0][0])

    # for i in range(2,len(contours)):
    #     hull = cv.convexHull(contours[i])
    #     hull_list.append(hull)

    crop_img = src[down:top, left:right] # Crop from {x, y, w, h } => {0, 0, 300, 400}
    
    x_offset=y_offset=60
    # l_img = cv.imread("qrcode.png")
    blank_image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    blank_image[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = crop_img


    cv.imwrite("test.jpg", blank_image)
    # # Draw hull results
    # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # color = (0, 256, 256)
    # cv.drawContours(drawing, hull_list, 0, color)
    # cv.imshow('Hull', drawing)


    # # image = cv2.resize(image,(int(new_dimensionX), int(new_dimensionY)))
    # # cv2.imwrite("test6.jpg", image)

    # # Draw contours results
    # # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    # # for i in range(len(contours)):
    # #     color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    # #     cv.drawContours(drawing, contours, i, color)
    # # Show in a window
    # cv.imshow('Contours', drawing)
# Load source image
parser = argparse.ArgumentParser(description='Code for Convex Hull tutorial.')
parser.add_argument('--input', help='Path to input image.', default='qr.png')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
cv.waitKey()

# WORK

# import cv2
# import numpy as np

# def thresh_callback(thresh):
#     edges = cv2.Canny(blur,thresh,thresh*2)
#     drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
#     contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         hull = cv2.convexHull(cnt)
#         cv2.drawContours(drawing,[cnt],0,(0,255,0),2)   # draw contours in green color
#         # cv2.drawContours(drawing,[hull],0,(0,0,255),2)  # draw contours in red color
#         cv2.imshow('output',drawing)
#         cv2.imshow('input',img)

# img = cv2.imread('qrcode.png')
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(5,5),0)

# cv2.namedWindow('input')

# thresh = 100
# max_thresh = 255

# cv2.createTrackbar('canny thresh:','input',thresh,max_thresh,thresh_callback)

# thresh_callback(0)

# if cv2.waitKey(0) == 27:
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     if(len(sys.argv)) < 2:
#         file_path = "qrcode.png"
#     else:
#         file_path = sys.argv[1]

#     # read image
#     src = cv2.imread(file_path, 1)
    
#     # show source image
#     cv2.imshow("Source", src)

#     # convert image to gray scale
#     gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
     
#     # blur the image
#     blur = cv2.blur(gray, (3, 3))
    
#     # binary thresholding of the image
#     ret, thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)
    
#     # find contours
#     im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
#     # create hull array for convexHull points
#     hull = []
    
#     # calculate points for each contour
#     for i in range(len(contours)):
#         hull.append(cv2.convexHull(contours[i], False))
    
#     # create an empty black image
#     drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    
#     # draw contours and hull points
#     for i in range(len(contours)):
#         color_contours = (0, 255, 0) # color for contours
#         color = (255, 255, 255) # color for convex hull
#         # draw contours
#         cv2.drawContours(drawing, contours, i, color_contours, 2, 8, hierarchy)
#         # draw convex hull
#         cv2.drawContours(drawing, hull, i, color, 2, 8)

#     cv2.imshow("Output", drawing)

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()