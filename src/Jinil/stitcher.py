import pdb
import glob
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from src.Jinil import utils
from src.Jinil import homographies



class PanaromaStitcher():
    def __init__(self):
        pass

    def homography_list(self, gt_images, mid_idx):
        
        H_list = []
        
        # From left to middle homography between idx-1 and idx
        for i in range(1, mid_idx+1):
            H_list.append(homographies.estimate_homography(gt_images[i - 1], gt_images[i]))
            
        # From right to middle homography between idx+1 and idx
        for i in range(mid_idx, len(gt_images)-1):
            H_list.append(homographies.estimate_homography(gt_images[i + 1], gt_images[i]))
            
        return H_list

    def image_stitching_left(self, img_src, img_des, H, shift_matrix_prev, total_weight):
        
        shift_matrix, size = utils.shift_matrix_left_size(img_src, img_des, H @ np.linalg.inv(shift_matrix_prev))
        shift_matrix = shift_matrix.astype(np.float64)
        
        warped_img_src = utils.warp_perpective(img_src, shift_matrix @ H @ np.linalg.inv(shift_matrix_prev), size)
        warped_img_des = utils.warp_perpective(img_des, shift_matrix, size)
        
        if total_weight is None:
            warped_weight_src = utils.blending_weight(img_src, shift_matrix @ H, size)
        else:
            warped_weight_src = cv2.warpPerspective(total_weight, shift_matrix @ H @ np.linalg.inv(shift_matrix_prev), size)
        
        warped_weight_des = utils.blending_weight(img_des, shift_matrix, size)
        
        weight_src_norm, weight_des_norm, total_weight_new = utils.normalize_weights(warped_weight_src, warped_weight_des)
        
        weight_src_norm = np.repeat(weight_src_norm[:, :, np.newaxis], 3, axis=2)
        weight_des_norm = np.repeat(weight_des_norm[:, :, np.newaxis], 3, axis=2)
        
        warped_img_src = warped_img_src.astype(np.float64)
        warped_img_des = warped_img_des.astype(np.float64)
        weight_src_norm = weight_src_norm.astype(np.float64)
        weight_des_norm = weight_des_norm.astype(np.float64)
        
        blended_img = (warped_img_src * weight_src_norm + warped_img_des * weight_des_norm)
        blended_img = np.clip(blended_img / blended_img.max() * 255, 0, 255).astype(np.uint8)
        
        return blended_img, shift_matrix, total_weight_new
    
    def image_stitching_right(self, img_src, img_des, H, shift_matrix_prev, total_weight):
        
        shift_matrix, size = utils.shift_matrix_right_size(img_src, img_des, H @ np.linalg.inv(shift_matrix_prev))
        shift_matrix = shift_matrix.astype(np.float64)
        
        warped_img_src = utils.warp_perpective(img_src, shift_matrix @ H @ np.linalg.inv(shift_matrix_prev), size)
        warped_img_des = utils.warp_perpective(img_des, shift_matrix, size)
        
        if total_weight is None:
            warped_weight_src = utils.blending_weight(img_src, shift_matrix @ H, size)
        else:
            warped_weight_src = cv2.warpPerspective(total_weight, shift_matrix @ H @ np.linalg.inv(shift_matrix_prev), size)
        
        warped_weight_des = utils.blending_weight(img_des, shift_matrix, size)
        
        weight_src_norm, weight_des_norm, total_weight_new = utils.normalize_weights(warped_weight_src, warped_weight_des)
        
        weight_src_norm = np.repeat(weight_src_norm[:, :, np.newaxis], 3, axis=2)
        weight_des_norm = np.repeat(weight_des_norm[:, :, np.newaxis], 3, axis=2)
        
        warped_img_src = warped_img_src.astype(np.float64)
        warped_img_des = warped_img_des.astype(np.float64)
        weight_src_norm = weight_src_norm.astype(np.float64)
        weight_des_norm = weight_des_norm.astype(np.float64)
        
        blended_img = (warped_img_src * weight_src_norm + warped_img_des * weight_des_norm)
        
        blended_img = np.clip(blended_img / blended_img.max() * 255, 0, 255).astype(np.uint8)
        
        return blended_img, shift_matrix, total_weight_new
    
    def make_panaroma_for_images_in(self,path):
        imf = path
        all_images = sorted(glob.glob(imf+os.sep+'*'))
        print('Found {} Images for stitching'.format(len(all_images)))
        
        # Read all images
        gt_images = [cv2.imread(img) for img in all_images]
        
        if gt_images[0].shape == (490, 653, 3) or gt_images[0].shape == (487, 730, 3):
            gt_images = [utils.cylindrical_warp(img, 800) for img in gt_images]
        
        # Find the middle image index
        if len(gt_images)%2 == 0:
            mid_idx = len(gt_images)//2 - 1
        else:
            mid_idx = len(gt_images)//2
            
        # Find homography list
        H_list = self.homography_list(gt_images, mid_idx)
        
        # Blending the left side of the middle image
        shift_matrix_left = np.eye(3)
        total_weight_left = None
        blended_img_left = gt_images[0].astype(np.float64)
        
        for i in range(1, mid_idx + 1):
            blended_img_left, shift_matrix_left, total_weight_left = self.image_stitching_left(blended_img_left, gt_images[i], H_list[i-1], shift_matrix_left, total_weight_left)
        
        # Blending the right side of the middle image
        shift_matrix_right = np.eye(3)
        total_weight_right = None
        blended_img_right = gt_images[-1].astype(np.float64)
        
        for i in range(len(gt_images) - 2, mid_idx, -1):
            blended_img_right, shift_matrix_right, total_weight_right = self.image_stitching_right(blended_img_right, gt_images[i], H_list[i], shift_matrix_right, total_weight_right)
            
        plt.imshow(cv2.cvtColor(blended_img_left, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()   
        
        plt.imshow(cv2.cvtColor(blended_img_right, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()   
        # Final stitching of the left and right blended images
        
        H = H_list[mid_idx]
        
        corners_left, min_x_left, min_y_left, max_x_left, max_y_left = utils.transform_corners(blended_img_left, np.linalg.inv(shift_matrix_left))
        corners_right, min_x_right, min_y_right, max_x_right, max_y_right = utils.transform_corners(blended_img_right, H @ np.linalg.inv(shift_matrix_right))
        
        canvas_width = int(max_x_right - min_x_left)
        canvas_height = int(max(max_y_left, max_y_right) - min(min_y_left, min_y_right))
        
        shift_matrix_final_left = np.array([[1, 0, -min_x_left], [0, 1, -min(min_y_left, min_y_right)], [0, 0, 1]])
        shift_matrix_final_right = np.array([[1, 0, (canvas_width -  max_x_right)], [0, 1, -min(min_y_left, min_y_right)], [0, 0, 1]])
        
        final_size = (canvas_width, canvas_height)
        
        warped_img_final_left = utils.warp_perpective(blended_img_left, shift_matrix_final_left @ np.linalg.inv(shift_matrix_left), final_size)
        warped_img_final_right = utils.warp_perpective(blended_img_right, shift_matrix_final_right @ H @ np.linalg.inv(shift_matrix_right), final_size)
        
        if total_weight_left is None:
            warped_weight_final_left = utils.blending_weight(blended_img_left, shift_matrix_final_left, final_size)
        else:
            warped_weight_final_left = cv2.warpPerspective(total_weight_left, shift_matrix_final_left @ np.linalg.inv(shift_matrix_left), final_size)
            
        if total_weight_right is None:
            warped_weight_final_right = utils.blending_weight(blended_img_right, shift_matrix_final_right @ H, final_size)
        else:
            warped_weight_final_right = cv2.warpPerspective(total_weight_right, shift_matrix_final_right @ H @ np.linalg.inv(shift_matrix_right), final_size)
            
        weight_final_left_norm, weight_final_right_norm, total_weight_final = utils.normalize_weights(warped_weight_final_left, warped_weight_final_right)
        
        weight_final_left_norm = np.repeat(weight_final_left_norm[:, :, np.newaxis], 3, axis=2)
        weight_final_right_norm = np.repeat(weight_final_right_norm[:, :, np.newaxis], 3, axis=2)
        
        plt.imshow(cv2.cvtColor(warped_img_final_left, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        plt.imshow(cv2.cvtColor(warped_img_final_right, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
        
        
        warped_img_final_left = warped_img_final_left.astype(np.float64)
        warped_img_final_right = warped_img_final_right.astype(np.float64)
        weight_final_left_norm = weight_final_left_norm.astype(np.float64)
        weight_final_right_norm = weight_final_right_norm.astype(np.float64)
        
        blended_img_final = (warped_img_final_left * weight_final_left_norm + warped_img_final_right * weight_final_right_norm)
        blended_img_final = np.clip(blended_img_final / blended_img_final.max() * 255, 0, 255).astype(np.uint8)
        
        return blended_img_final, H_list
                

