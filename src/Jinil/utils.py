import numpy as np
from PIL import Image
import cv2

def bilinear_interpolation(x, y, img):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    
    x0 = np.clip(x0, 0, img.shape[1]-1)
    x1 = np.clip(x1, 0, img.shape[1]-1)
    y0 = np.clip(y0, 0, img.shape[0]-1)
    y1 = np.clip(y1, 0, img.shape[0]-1)
    
    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]
    
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    return Ia*wa[:, None] + Ib*wb[:, None] + Ic*wc[:, None] + Id*wd[:, None]

def transform(src_pts, H):
    src = np.pad(src_pts, ((0, 0), (0, 1)), constant_values=1)
    des = np.dot(H, src.T).T
    des = (des / des[:, 2].reshape(-1, 1))[:, :2]
    return des

def cylindrical_warp(img, focal_length):
    '''
    Warp img to a cylindrical coordinate system.
    '''
    h, w = img.shape[:2]
    x_c, y_c = w // 2, h // 2
    
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    theta = (u - x_c) / focal_length
    h_cyl = (v - y_c) / focal_length
    
    x_cyl = np.sin(theta)
    y_cyl = h_cyl
    z_cyl = np.cos(theta)
    
    # Convert from cylindrical coordinates to image coordinates
    x_img = np.round(focal_length * x_cyl / z_cyl + x_c).astype(np.int32)
    y_img = np.round(focal_length * y_cyl / z_cyl + y_c).astype(np.int32)
    
    # Create a mask for valid points within the image bounds
    mask = (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)
    
    # Create the output image, initially black
    cylindrical_img = np.zeros_like(img)
    cylindrical_img[v[mask], u[mask]] = img[y_img[mask], x_img[mask]]
    
    cylindrical_img = Image.fromarray(cylindrical_img)
    cylindrical_img = cylindrical_img.crop((u[mask].min(), v[mask].min(), u[mask].max(), v[mask].max()))
    cylindrical_img = np.array(cylindrical_img)
    
    return cylindrical_img

def warp_perpective(img, H, size):
    width, height = size

    warped_img = np.zeros((height, width, img.shape[2]), dtype=np.uint8)
    idx_pts = np.mgrid[0:width, 0:height].reshape(2, -1).T
    
    map_pts = transform(idx_pts, np.linalg.inv(H))
    x, y = map_pts[:, 0], map_pts[:, 1]
    
    mask = (x >= 0) & (x < img.shape[1]) & (y >= 0) & (y < img.shape[0])
    
    valid_x, valid_y = x[mask], y[mask]
    idx_pts = idx_pts[mask]
    
    warped = bilinear_interpolation(valid_x, valid_y, img)
    
    warped_img[idx_pts[:, 1], idx_pts[:, 0]] = warped
    return warped_img

def single_weights_array(size: int) -> np.ndarray:
    if size % 2 == 1:
        return np.concatenate([np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]])
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])


def single_weights_matrix(shape: tuple[int]) -> np.ndarray:
    return (single_weights_array(shape[0])[:, np.newaxis] @ single_weights_array(shape[1])[:, np.newaxis].T)


def blending_weight(img, sift_matrix, canvas):
    weight = single_weights_matrix(img.shape[:2])
    warped_weight = cv2.warpPerspective(weight, sift_matrix, canvas)
    return warped_weight
    
def normalize_weights(weight1, weight2):
    total_weight = (weight1 + weight2) / (weight1 + weight2).max()
    
    weight1_norm = np.divide(weight1, total_weight, where=total_weight!=0)
    weight2_norm = np.divide(weight2, total_weight, where=total_weight!=0)
    
    return weight1_norm, weight2_norm, total_weight

def transform_corners(img, sift_matrix):
    h, w = img.shape[:2]
    
    corners = np.array([[0, 0], [0, h-1], [w-1, 0], [w-1, h-1]], dtype=np.float32)
    transformed_corners = transform(corners, sift_matrix)
    transform_corners = transformed_corners.reshape(4, 2)
    
    min_x, min_y = np.min(transform_corners, axis=0)
    max_x, max_y = np.max(transform_corners, axis=0)
    
    return transformed_corners, min_x, min_y, max_x, max_y

def shift_matrix_left_size(img_src, img_des, H):
    
    corners, min_x, min_y, max_x, max_y = transform_corners(img_src, H)
    shift_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    size = img_des.shape[1] - int(min_x), img_des.shape[0] - int(min_y)
    
    return shift_matrix, size

def shift_matrix_right_size(img_src, img_des, H):
        
        corners, min_x, min_y, max_x, max_y = transform_corners(img_src, H)
        shift_x, shift_y = -min(min_x, 0), -min(min_y, 0)
        shift_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])
        size = int(max(img_des.shape[1], int(max_x)) + shift_x), int(max(img_des.shape[0], int(max_y)) + shift_y)
        
        return shift_matrix, size
       