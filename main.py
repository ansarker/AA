import cv2
import time
import numpy as np
from scipy.spatial.distance import cdist
import math
import random

from lib.interfaces import Mesh
from lib.mc.mc import TriangleMeshCreator
from lib.md.deform import ARAPDeformation
from utils import image_utils as im_utils
from utils import object_utils as obj_utils
from utils import render_utils
from utils.shape_context import ShapeContext
from utils.tps import TPS
import json


VISUALIZE = True
DEFORM_MESH_PATH = './deformed_mesh.obj'


def augment_handle_points(poses2d, size):
    target_poses2d = poses2d.copy()
    target_poses2d[5] = [100, 150]
    return target_poses2d

def augment_handle_points_(lines, size):
    target_poses2d = np.zeros((len(lines), 2), dtype=np.int32)
    
    for i in range(len(lines)):
        target_poses2d[i] = lines[i][1]
    
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
    target_model_path = "./data/osim.jpg"
    target_mask_path = "./data/osim.png"
    clothes_path = "./data/purple.png"
    poses2d_path = "./data/osim_keypoints.json"

    target_model_image = cv2.imread(target_model_path)
    # target_model_image = cv2.cvtColor(target_model_image, cv2.COLOR_BGR2RGB)
    h, w = target_model_image.shape[:2]
    target_clothes_mask = cv2.imread(target_mask_path, 0)
    # target_clothes_mask = cv2.GaussianBlur(target_clothes_mask, (5, 5), 0)
    # target_clothes_mask = cv2.Canny(target_clothes_mask, 50, 200)
    target_clothes_mask_contour, _ = cv2.findContours(target_clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_clothes_mask_contour = target_clothes_mask_contour[0]
    target_clothes_mask_contour_sq = target_clothes_mask_contour.squeeze()

    clothes = cv2.imread(clothes_path)
    clothes_gray = cv2.cvtColor(clothes, cv2.COLOR_BGR2GRAY)
    # clothes_gray = cv2.GaussianBlur(clothes_gray, (5, 5), 0)
    # clothes_gray = cv2.Canny(clothes_gray, 50, 200)
    clothes_mask = cv2.threshold(clothes_gray, 0, 255, cv2.THRESH_BINARY)[1]
    clothes_mask_contour, _ = cv2.findContours(clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clothes_mask_contour = clothes_mask_contour[0]
    clothes_mask_contour_sq = clothes_mask_contour.squeeze()
    
    # Bounding box for the target and source contours
    target_x, target_y, target_w, target_h = cv2.boundingRect(target_clothes_mask_contour)
    source_x, source_y, source_w, source_h = cv2.boundingRect(clothes_mask_contour)
    
    mask_box = target_clothes_mask[target_y:target_y+target_h, target_x:target_x+target_w]
    clothes_mask_box = clothes_mask[source_y:source_y+source_h, source_x:source_x+source_w]
    clothes_mask_box = cv2.resize(clothes_mask_box, (target_w, target_h))
    
    # cv2.imshow("target clothes mask", mask_box)
    # cv2.imshow("clothes mask", clothes_mask_box)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    sc = ShapeContext()
    sampls = 100
    rotate = False
    
    # target_clothes_mask contour and clothes contour
    mask_contour, _ = cv2.findContours(target_clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_contour = mask_contour[0]
    mask_contour_sq = mask_contour.squeeze()
    
    clothes_mask_contour, _ = cv2.findContours(clothes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clothes_mask_contour = clothes_mask_contour[0]
    clothes_mask_contour_sq = clothes_mask_contour.squeeze()
    
    # Bounding box for the target and source contours
    target_x, target_y, target_w, target_h = cv2.boundingRect(mask_contour)
    source_x, source_y, source_w, source_h = cv2.boundingRect(clothes_mask_contour)
    
    target_image_box = target_model_image[target_y:target_y+target_h, target_x:target_x+target_w]
    mask_box = target_clothes_mask[target_y:target_y+target_h, target_x:target_x+target_w]
    clothes_mask_box = clothes_mask[source_y:source_y+source_h, source_x:source_x+source_w]
    clothes_mask_box = cv2.resize(clothes_mask_box, (target_w, target_h))
    # clothes_mask_box = cv2.GaussianBlur(clothes_mask_box, (5, 5), 0)
    # clothes_mask_box = cv2.Canny(clothes_mask_box, 50, 200)
    
    # original target_model_image resize
    clothes_box = clothes[source_y:source_y+source_h, source_x:source_x+source_w]
    clothes_box = cv2.resize(clothes_box, (target_w, target_h))
    # clothes_box = cv2.GaussianBlur(clothes_box, (5, 5), 0)
    # clothes_box = cv2.Canny(clothes_box, 50, 200)
    

    points1, t1 = sc.get_points_from_img(mask_box, simpleto=sampls)
    points2, t2 = sc.get_points_from_img(clothes_mask_box, simpleto=sampls)

    if rotate:
        # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
        theta = np.radians(90)
        c, s = np.cos(theta), np.sin(theta)
        R = np.matrix('{} {}; {} {}'.format(c, -s, s, c))
        points2 = np.dot(np.array(points2), R).tolist()

    P = sc.compute(points1)
    x1 = [p[1] for p in points1]
    y1 = [p[0] for p in points1]
    Q = sc.compute(points2)
    x2 = [p[1] for p in points2]
    y2 = [p[0] for p in points2]

    standard_cost,indexes = sc.diff(P,Q)

    lines = []
    matched_values = []
    for p,q in indexes:
        distance = math.sqrt(((points1[p][0]-points2[q][0])**2)+((points1[p][1]-points2[q][1])**2))
        if distance < standard_cost:
            lines.append(((points1[p][1],points1[p][0]), (points2[q][1],points2[q][0])))
            matched_values.append((points1[p][1],points1[p][0]))
    
    out_img = target_model_image.copy()
    out_img_box = out_img[target_y:target_y+target_h, target_x:target_x+target_w]
    
    # Create 3-channel target_clothes_mask of float datatype
    alpha = cv2.cvtColor(clothes_mask_box, cv2.COLOR_GRAY2BGR)/255.0

    # Perform blending and limit pixel values to 0-255
    blended = cv2.convertScaleAbs(out_img_box * (1-alpha) + clothes_box * alpha)
    
    # out_img[target_y:target_y+target_h, target_x:target_x+target_w] = clothes_box
    out_img[target_y:target_y+target_h, target_x:target_x+target_w] = blended
    
    poses2d = np.array(matched_values)

    tri_mc = TriangleMeshCreator(interval=20, angle_constraint=15, area_constraint=200, dilated_pixel=5)
    mesh = tri_mc.create(target_model_image, target_clothes_mask)

    vertices = 0.5 * (mesh.vertices + 1) * np.array([w, h]).reshape((1, 2)).astype(np.float32)
    distance = cdist(poses2d, vertices)
    constraint_v_ids = np.argmin(distance, axis=1)
    poses2d = vertices[constraint_v_ids]
    
    constraint_v_coords = augment_handle_points_(lines, size=(w, h))

    # constraint_v_ids = np.array([e for i, e in enumerate(constraint_v_ids) if i != 3])
    # constraint_v_coords = np.array([e for i, e in enumerate(constraint_v_coords) if i != 3])

    l2 = []
    if VISUALIZE:
        vis_image = mesh.get_image()
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        black = target_image_box.copy()
        
        for i, (x, y) in enumerate(lines):
            cv2.circle(black, x, radius=3, color=(255, 0, 0), thickness=1)
            l2.append(x)

        for x, y in constraint_v_coords.astype(np.int32):
            cv2.circle(black, (x, y), radius=3, color=(0, 255, 0), thickness=1)
            
        for i in range(len(lines)):
            cv2.line(black, lines[i][0], constraint_v_coords[i].astype(np.int32), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)), 1)

        # im_utils.imshow(black)
        # cv2.imshow('target box', black)
        # cv2.imshow('cloth box', clothes_box)
    
    # new_image, matches = obj_utils.WarpImage_TPS(source=constraint_v_coords, target=np.array(l2), img=clothes)
    # new_image = obj_utils.tps_trans(p1=constraint_v_coords, p2=np.array(l2), gray=clothes)
    target_model_image = cv2.cvtColor(target_model_image, cv2.COLOR_RGB2RGBA)
    new_image = obj_utils.warp(img1=clothes_box, img2=target_image_box, pts1=constraint_v_coords, pts2=np.array(l2))
    
    target_model_image[target_y:target_y+target_h, target_x:target_x+target_w] = new_image
    
    cv2.imshow('new_image', target_model_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
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
