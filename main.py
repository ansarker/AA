import cv2
import time
import numpy as np
from scipy.spatial.distance import cdist

from lib.interfaces import Mesh
from lib.mc.mc import TriangleMeshCreator
from lib.md.deform import ARAPDeformation
from utils import image_utils as im_utils
from utils import object_utils as obj_utils
from utils import render_utils
import json


VISUALIZE = True
DEFORM_MESH_PATH = './deformed_mesh.obj'


def augment_handle_points(poses2d, size):
    target_poses2d = poses2d.copy()
    target_poses2d[5] = [100, 150]
    return target_poses2d

def augment_handle_points_(from_poses2d, to_poses2d, size):
    target_poses2d = to_poses2d.copy()
    source_poses2d = from_poses2d.copy()
    
    for i in range(target_poses2d.shape[0]):
        target_poses2d[i] = source_poses2d[i]
        # print(i, " >-> ", source_poses2d[i], " >-> ", target_poses2d[i])

    return target_poses2d

def save_obj_format(file_path, vertices, faces, texture_vertices=None):
    """
    Save obj wavefront to file
    :param file_path:
    :param vertices: in range [-1.,1.]
    :param faces:
    :param texture_vertices
    :return:
    """

    f = open(file_path, 'w')

    # number of vertices
    no_v = len(vertices)
    no_f = len(faces)

    f.write('#vertices: %d\n' % no_v)
    f.write('#faces: %d\n' % no_f)

    # vertices
    for i in range(no_v):
        v = vertices[i]
        f.write("v %.4f %.4f %d\n" % (v[0], v[1], 0))

    # vertices texture
    if texture_vertices is not None:
        # the origin of texture vertices are not TOP-LEFT, but BOT-LEFT
        for i in range(no_v):
            v = texture_vertices[i]
            f.write("vt %.4f %.4f\n" % (v[0], v[1]))

    # triangle faces
    for t in faces:
        f.write("f")
        for i in t:
            f.write(" %d/%d" % (i,i))
        f.write("\n")

    f.close()


def main():
    # image_path = "./data/image_alok.png"
    # mask_path = "./data/mask_alok.png"
    # poses2d_path = "./data/poses2d.npy"
    image_path = "./data/osim.jpg"
    mask_path = "./data/osim.png"
    clothes_path = "./data/align_shirt.png"
    poses2d_path = "./data/osim_keypoints.json"

    #
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    mask = cv2.imread(mask_path, 0)
    clothes = cv2.imread(clothes_path)
    clothes_gray = cv2.cvtColor(clothes, cv2.COLOR_BGR2GRAY)
    clothes_mask = cv2.threshold(clothes_gray, 0, 255, cv2.THRESH_BINARY)[1]
    
    # mask contour and clothes contour
    mask_contour, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = mask_contour[0]
    mask_contour_sq = mask_contour.squeeze()
    
    clothes_mask_contour, _ = cv2.findContours(clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clothes_mask_contour = clothes_mask_contour[0]
    clothes_mask_contour_sq = clothes_mask_contour.squeeze()
    
    # Bounding box for the target and source contours
    target_x, target_y, target_w, target_h = cv2.boundingRect(mask_contour)
    source_x, source_y, source_w, source_h = cv2.boundingRect(clothes_mask_contour)
    
    mask_box = mask[target_y:target_y+target_h, target_x:target_x+target_w]
    clothes_mask_box = clothes_mask[source_y:source_y+source_h, source_x:source_x+source_w]
    clothes_mask_box = cv2.resize(clothes_mask_box, (target_w, target_h))
    
    # original image resize
    clothes_box = clothes[source_y:source_y+source_h, source_x:source_x+source_w]
    clothes_box = cv2.resize(clothes_box, (target_w, target_h))
    
    # Draw contours
    mask_contour_img = cv2.drawContours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), mask_contour, -1, (0, 255, 0), 2)
    clothes_mask_contour_img = cv2.drawContours(cv2.cvtColor(clothes_mask, cv2.COLOR_GRAY2BGR), clothes_mask_contour, -1, (0, 255, 0), 2)

    out_img = image.copy()
    out_img_box = out_img[target_y:target_y+target_h, target_x:target_x+target_w]
    
    # Create 3-channel mask of float datatype
    alpha = cv2.cvtColor(clothes_mask_box, cv2.COLOR_GRAY2BGR)/255.0

    # Perform blending and limit pixel values to 0-255
    blended = cv2.convertScaleAbs(out_img_box * (1-alpha) + clothes_box * alpha)
    
    # out_img[target_y:target_y+target_h, target_x:target_x+target_w] = clothes_box
    out_img[target_y:target_y+target_h, target_x:target_x+target_w] = blended
    
    # compare two contours and get the matched values
    threshold = 100
    matched_values, matched_indices = obj_utils.compare_arrays(clothes_mask_contour, mask_contour, threshold)
    matched_values = np.expand_dims(matched_values, axis=1)

    # cv2.imshow('mask', mask_box)
    # cv2.imshow('clothes', clothes_mask_box)
    # cv2.imshow('clothes contour', clothes_mask_contour_img)
    cv2.imshow('Out image', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # poses2d = np.load(poses2d_path)
    
    # with open(poses2d_path) as f:
    #     pose_label = json.load(f)
    #     poses2d = pose_label['people'][0]['pose_keypoints_2d']
    #     poses2d = np.array(poses2d).astype(np.int32)
    #     poses2d = poses2d.reshape((-1, 3))[:, :2]

    poses2d = matched_values.squeeze()
    
    #
    tri_mc = TriangleMeshCreator(interval=20, angle_constraint=15, area_constraint=200, dilated_pixel=5)
    mesh = tri_mc.create(image, mask)

    #
    vertices = 0.5 * (mesh.vertices + 1) * np.array([w, h]).reshape((1, 2)).astype(np.float32)
    distance = cdist(poses2d, vertices)
    constraint_v_ids = np.argmin(distance, axis=1)
    poses2d = vertices[constraint_v_ids]
    # constraint_v_coords = augment_handle_points(poses2d, size=(w, h))
    constraint_v_coords = augment_handle_points_(clothes_mask_contour_sq, poses2d, size=(w, h))

    # constraint_v_ids = np.array([e for i, e in enumerate(constraint_v_ids) if i != 3])
    # constraint_v_coords = np.array([e for i, e in enumerate(constraint_v_coords) if i != 3])

    if VISUALIZE:
        vis_image = mesh.get_image()
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        for x, y in poses2d.astype(np.int32):
            cv2.circle(vis_image, (x, y), radius=3, color=(255, 0, 0), thickness=2)

        for x, y in constraint_v_coords.astype(np.int32):
            cv2.circle(vis_image, (x, y), radius=3, color=(0, 255, 0), thickness=2)

        im_utils.imshow(vis_image)

    #
    constraint_v_coords = Mesh.normalize_vertices(constraint_v_coords, size=(w, h))

    # build vertices texture
    vts = 0.5 * (mesh.vertices + 1)
    vts[:, 1] = 1. - vts[:, 1]

    # deform
    arap_deform = ARAPDeformation()
    arap_deform.load_from_mesh(mesh)
    arap_deform.setup()

    deformed_mesh = arap_deform.deform(constraint_v_ids, constraint_v_coords, w=1000.)
    save_obj_format(file_path=DEFORM_MESH_PATH, vertices=deformed_mesh.vertices, faces=deformed_mesh.faces,
                    texture_vertices=vts)

    if VISUALIZE:
        vis_image = deformed_mesh.get_image(size=(w, h))
        im_utils.imshow(vis_image)

    #
    pt_renderer = render_utils.PytorchRenderer(use_gpu=False)
    deformed_image = pt_renderer.render_w_texture(DEFORM_MESH_PATH, clothes_path)
    deformed_image = deformed_image[::-1, :, :]
    deformed_image = cv2.cvtColor(deformed_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite('./images/deformed.jpg', cv2.cvtColor(deformed_image, cv2.COLOR_BGR2RGB))
    im_utils.imshow(deformed_image)


if __name__ == '__main__':
    main()
