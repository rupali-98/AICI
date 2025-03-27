#### Importing Libraries ###
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def noise_reduction(img):
    blurred = cv2.medianBlur(img, 3) ## Removal of Salt and pepper noise
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)  # Create binary image.
    # kernel = np.ones((2, 2), np.uint8)  # Define the structuring element
    # closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2) # Apply Closing
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.morphologyEx(blurred, cv2.MORPH_ERODE, kernel, iterations=2)  # Apply erosion
    return eroded

def line_detection(img):
    high_thresh, thresh_im = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    ## 1. Canny Edge Detector
    edges = cv2.Canny(img, lowThresh, high_thresh)
    kernel = np.ones((2, 2), np.uint8)
    ## 2. Dilating the edges
    morph = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel) ## Dilate the edges
    # 3. Hough Transform with Adjusted Parameters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lines = cv2.HoughLinesP(morph, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=15)  # Adjusted parameters
    # 4. Line Filtering (More Reasonable Length)
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > 10:  # More reasonable length threshold
                filtered_lines.append(line[0])

    # 5 Create Floor Plan Image
    floor_plan = np.zeros_like(gray)
    if filtered_lines:
        for line in filtered_lines:
            x1, y1, x2, y2 = line
            cv2.line(floor_plan, (x1, y1), (x2, y2), 255, 2)
    return floor_plan

def morph_operation_to_get_line(img):
    # Morphological Operations
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    morph2 = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel,iterations=2)
    return morph2

# def contouring(combined_lines,morph):
#     # gray_morph = cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)
#     cnts1 = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     cnts = cnts1[0]
#     # Filter contours by area
#     min_area = 1000  # Adjust as needed
#     filtered_contours = [c for c in cnts if cv2.contourArea(c) > min_area]

#     for c in filtered_contours:
#         left = tuple(c[c[:, :, 0].argmin()][0])
#         right = tuple(c[c[:, :, 0].argmax()][0])
#         top = tuple(c[c[:, :, 1].argmin()][0])
#         bottom = tuple(c[c[:, :, 1].argmax()][0])

#     for i in range(len(cnts)):

#         min_dist = max(combined_lines.shape[0], combined_lines.shape[1])

#         cl = []

#         ci = cnts[i]
#         ci_left = tuple(ci[ci[:, :, 0].argmin()][0])
#         ci_right = tuple(ci[ci[:, :, 0].argmax()][0])
#         ci_top = tuple(ci[ci[:, :, 1].argmin()][0])
#         ci_bottom = tuple(ci[ci[:, :, 1].argmax()][0])
#         ci_list = [ci_bottom, ci_left, ci_right, ci_top]

#         for j in range(i + 1, len(cnts)):
#             cj = cnts[j]
#             cj_left = tuple(cj[cj[:, :, 0].argmin()][0])
#             cj_right = tuple(cj[cj[:, :, 0].argmax()][0])
#             cj_top = tuple(cj[cj[:, :, 1].argmin()][0])
#             cj_bottom = tuple(cj[cj[:, :, 1].argmax()][0])
#             cj_list = [cj_bottom, cj_left, cj_right, cj_top]

#             for pt1 in ci_list:
#               for pt2 in cj_list:
#                 dist = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
#                 if dist < min_dist:
#                   min_dist = dist
#                   cl = []
#                   cl.append([pt1, pt2, min_dist])

#         if len(cl) > 0:
#           cv2.line(morph, cl[0][0], cl[0][1], (255, 255, 255), thickness=10)
#     return morph

def refine_boundary(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    refined_img = np.zeros_like(img)
    for contour in contours:
        # Check if the contour is closed (area > 0)
        if cv2.contourArea(contour) > 1000:
            # Approximate the contour with a polygon
            epsilon = 0.0005 * cv2.arcLength(contour, True)  # Adjust epsilon for accuracy
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(refined_img, [approx], 0, 255, thickness=2)

    return refined_img
    # Create a blank image to draw the refined boundaries
