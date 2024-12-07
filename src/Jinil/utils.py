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
    warped_weight = warp_perpective(weight, sift_matrix, canvas)
    return warped_weight
    
def normalize_weights(weight1, weight2):
    total_weight = (weight1 + weight2) / (weight1 + weight2).max()
    
    weight1_norm = np.devide(weight1, total_weight, where=total_weight!=0)
    weight2_norm = np.devide(weight2, total_weight, where=total_weight!=0)
    
    return weight1_norm, weight2_norm, total_weight

def transform_corners(img, sift_matrix):
    h, w = img.shape[:2]
    
    corners = np.array([[0, 0], [0, h-1], [w-1, 0], [w-1, h-1]], dtype=np.float32)
    transformed_corners = transform(corners, sift_matrix)
    transform_corners = transformed_corners.reshape(4, 2)
    
    min_x, min_y = np.min(transform_corners, axis=0)
    max_x, max_y = np.max(transform_corners, axis=0)
    
    return transformed_corners, min_x, min_y, max_x, max_y

def shift_matrix_left_size(img, img_ref, H):
    
    corners, min_x, min_y, max_x, max_y = transform_corners(img, H)
    shift_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    size = img_ref.shape[1] - int(min_x), img_ref.shape[0] - int(min_y)
    
    return shift_matrix, size

def shift_matrix_right_size(img, img_ref, H):
        
        corners, min_x, min_y, max_x, max_y = transform_corners(img, H)
        shift_x, shift_y = -min(min_x, 0), -min(min_y, 0)
        shift_matrix = np.array([[1, 0, shift_x], [0, 1, shift_y], [0, 0, 1]])
        size = int(max(img_ref.shape[1], int(max_x)) + shift_x), int(max(img_ref.shape[0], int(max_y)) + shift_y)
        
        return shift_matrix, size
       