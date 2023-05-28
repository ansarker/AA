import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt
import numpy as np
import random

def find_nearest_neighbors(source_array, target_array):
    # Flatten the arrays to 2 dimensions for nearest neighbor search
    source_flattened = source_array.reshape(-1, 2)
    target_flattened = target_array.reshape(-1, 2)
    
    tree = cKDTree(target_flattened)
    distances, indices = tree.query(source_flattened)
    return distances, indices

def compare_arrays(source_array, target_array, threshold):
    black = np.zeros((512, 384, 3), dtype=np.uint8)
    black1 = black.copy()
    
    distances, indices = find_nearest_neighbors(source_array, target_array)
    matched_indices = np.where(distances <= threshold)[0]
    matched_values = target_array.reshape(-1, 2)[indices[matched_indices]]
    
    # Visualize the matched points
    # cv2.drawContours(black, np.expand_dims(matched_values, axis=1), -1, (255, 0, 0), 2)
    # cv2.drawContours(black, np.expand_dims(source_array, axis=1), -1, (0, 255, 0), 2)
    # for i in range(len(matched_indices)):
    #     cv2.line(black1, tuple(source_array[i][0]), tuple(matched_values[i]), (136, 200, 85), 1)
    # cv2.imshow('matched', black)
    # cv2.imshow('matched1', black1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return matched_values, matched_indices

def tps_trans(p1,p2,gray,tps_lambda = 0.2):
    '''
    Thin-Plate Spline Transform Algorithm 

    input:
        p1 : feature points in the image to be transformed
        p2 : target feature points
        gray : input image
        tps_lambda : a tps parameter
    output:
        out_img : transformed input image
        new : transformed mark image, where shows transformed p1 points 
                (as cv2.applyTransformation() cannot work properly)
    '''
    p1 = p1.reshape(-1, len(p1), 2)
    p2 = p2.reshape(-1, len(p2), 2)
    
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.setRegularizationParameter(tps_lambda)
    matches = []
    new = np.zeros_like(gray)
    for i in range(0,len(p1[0])):
        matches.append(cv2.DMatch(i,i,0))
        # cv2.circle(new,[int(p1[0][i][0]),int(p1[0][i][1])],1,(1,0,0),-1)
    
    tps.estimateTransformation(p2, p1, matches)

    out_img = tps.warpImage(gray)
    print(tps.applyTransformation(p1))
    return out_img


def triangles(points):
    points = np.where(points, points, 1)
    subdiv = cv2.Subdiv2D((*points.min(0), *points.max(0)))
    for pt in points:
        subdiv.insert(tuple(map(int, pt)))
    for pts in subdiv.getTriangleList().reshape(-1, 3, 2):
        yield [np.where(np.all(points == pt, 1))[0][0] for pt in pts]

def crop(img, pts):
    x, y, w, h = cv2.boundingRect(pts)
    img_cropped = img[y: y + h, x: x + w]
    pts[:, 0] -= x
    pts[:, 1] -= y
    return img_cropped, pts

def warp(img1, img2, pts1, pts2):
    img2 = np.zeros_like(img1)

    for indices in triangles(pts1):
        img1_cropped, triangle1 = crop(img1, pts1[indices])
        img2_cropped, triangle2 = crop(img2, pts2[indices])
        transform = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
        img2_warped = cv2.warpAffine(img1_cropped, transform, img2_cropped.shape[:2][::-1], None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
        mask = np.zeros_like(img2_cropped)
        cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 50, 0)
        img2_cropped *= 1 - mask
        img2_cropped += img2_warped * mask
    
    return img2

def remove_fringe(img):
    h, w = img.shape[:2]

    # create zeros mask 2 pixels larger in each dimension
    mask = np.zeros([h + 2, w + 2], np.uint8)

    # floodfill background with black
    floodfill = cv2.floodFill(img, mask, (0,0), (0,0,0), (1), (0), flags=8)[1]

    # make all other colors but black into white
    mask = floodfill.copy()
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask!=0] = 255

    # erode mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    # use mask to change background to white
    result = img.copy()
    result[mask==0] = 255
    
    # save cropped image
    cv2.imwrite('./cl/food_floodfilled.jpg',floodfill)
    cv2.imwrite('./cl/food_mask.jpg',mask)
    cv2.imwrite('./cl/food_processed.jpg',result)

    # show the images
    #cv2.imshow("thresh", thresh)
    cv2.imshow("floodfill", floodfill)
    cv2.imshow("mask", mask)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result

def max_contour(contours):
    if len(contours) != 0:
        # find the biggest countour (c) by the area
        max_c = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_c)

        return max_c, (x, y, w, h)

if __name__ == "__main__":
    target_mask = cv2.imread('data/osim.png', cv2.IMREAD_GRAYSCALE)
    source = cv2.imread('data/align_shirt.png', cv2.IMREAD_GRAYSCALE)
    source_mask = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY)[1]
            
    # Find contours in the source and target masks
    target_contour, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    source_contour, _ = cv2.findContours(source_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    target_contour = target_contour[0]
    source_contour = source_contour[0]
    
    # target_contour = resample_contour(target_contour, 20)
    # print(target_contour)
    print(target_contour.shape)
    print(source_contour.shape)
    
    threshold = 30
    matched_values, matched_indices = compare_arrays(source_contour, target_contour, threshold)
    matched_values = np.expand_dims(matched_values, axis=1)

    # Bounding box for the target and source contours
    target_x, target_y, target_w, target_h = cv2.boundingRect(target_contour)
    source_x, source_y, source_w, source_h = cv2.boundingRect(source_contour)
    
    target_mask_box = target_mask[target_y:target_y+target_h, target_x:target_x+target_w]
    source_mask_box = source_mask[source_y:source_y+source_h, source_x:source_x+source_w]
    
    # Resize contours to have the same number of points
    source_mask_box = cv2.resize(source_mask_box, (target_w, target_h))
    
    # Difference between two images
    diff = cv2.absdiff(target_mask_box, source_mask_box)
    
    # Visualize the contours
    source_contour_vis = cv2.drawContours(np.zeros_like(source_mask), source_contour, -1, (255, 0, 0), 2)
    target_contour_vis = cv2.drawContours(np.zeros_like(target_mask), target_contour, -1, (255, 0, 0), 2)
        
    # Crop contours area
    source_contour_box = source_contour_vis[source_y:source_y+source_h, source_x:source_x+source_w]
    target_contour_box = target_contour_vis[target_y:target_y+target_h, target_x:target_x+target_w]
    
    # Resize contours to have the same number of points
    source_contour_box = cv2.resize(source_contour_box, (target_w, target_h))
    
    # Difference between two contours
    contour_diff = cv2.absdiff(target_contour_box, source_contour_box)
    
    # Matched contours
    matched_contour = np.zeros_like(source_mask)
    cv2.drawContours(matched_contour, matched_values, -1, (255, 0, 0), 2)
    matched_contour = matched_contour[target_y:target_y+target_h, target_x:target_x+target_w]
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 15))
    axs[0, 0].imshow(source_mask_box, cmap='gray')
    axs[0, 0].set_title('Source')
    axs[0, 1].imshow(target_mask_box, cmap='gray')
    axs[0, 1].set_title('Target')
    axs[0, 2].imshow(diff, cmap='gray')
    axs[0, 2].set_title('Difference')
    
    axs[1, 0].imshow(source_contour_box, cmap='gray')
    axs[1, 0].set_title('Source Contour')
    axs[1, 1].imshow(target_contour_box, cmap='gray')
    axs[1, 1].set_title('Target Contour')
    # axs[1, 2].imshow(contour_diff, cmap='gray')
    # axs[1, 2].set_title('Contour Difference')
    axs[1, 2].imshow(matched_contour, cmap='gray')
    axs[1, 2].set_title('Matched')
    plt.show()
