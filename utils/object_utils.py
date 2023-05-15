import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import svd
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
    
    print(target_contour.shape)
    print(source_contour.shape)
    
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
    axs[1, 2].imshow(contour_diff, cmap='gray')
    axs[1, 2].set_title('Contour Difference')
    plt.show()
