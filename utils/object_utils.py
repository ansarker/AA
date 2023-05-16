import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

import numpy as np

def procrustes_analysis(source_points, target_points):
    # Convert to float data type
    source_points = source_points.astype(np.float64)
    target_points = target_points.astype(np.float64)

    # Center the points
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_points -= source_centroid
    target_points -= target_centroid

    # Scale the points
    source_norm = np.linalg.norm(source_points)
    target_norm = np.linalg.norm(target_points)
    source_points /= source_norm
    target_points /= target_norm

    # Compute the rotation matrix
    u, _, vt = np.linalg.svd(np.dot(source_points.T, target_points))
    R = np.dot(u, vt)

    # Apply the transformation to the source points
    transformed_points = np.dot(source_points, R) + target_centroid

    return transformed_points

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
    cv2.drawContours(black, np.expand_dims(matched_values, axis=1), -1, (255, 0, 0), 2)
    cv2.drawContours(black, np.expand_dims(source_array, axis=1), -1, (0, 255, 0), 2)
    for i in range(len(matched_indices)):
        cv2.line(black1, tuple(source_array[i][0]), tuple(matched_values[i]), (136, 200, 85), 1)
    cv2.imshow('matched', black)
    cv2.imshow('matched1', black1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return matched_values, matched_indices

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
