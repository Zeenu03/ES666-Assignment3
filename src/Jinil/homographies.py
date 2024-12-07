import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_keypoints_discriptors(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    return keypoints1, descriptors1, keypoints2, descriptors2

def find_matches(keypoints1, keypoints2):
    matches = []

    for i in range(len(keypoints1)):
        distances = np.linalg.norm(keypoints2 - keypoints1[i], axis=1)
        m = np.argmin(distances)
        n = np.argsort(distances)[1]
        if distances[m] < 0.75 * distances[n]:
            matches.append([i, m])
    return np.array(matches)

def compute_homography(matches, keypoints_src, keypoints_des):
    A = []

    for i, j in matches:
        x1, y1 = keypoints_src[i].pt
        x2, y2 = keypoints_des[j].pt

        # Populate A matrix for the homogeneous transformation
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])

    # Convert A to numpy array
    A = np.array(A)

    # Solve Ah = 0 using SVD
    _, _, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)

    # Normalize H to make the bottom-right value 1
    H = H / H[2, 2]
    return H

def RANSAC_homography(matches, keypoints1, keypoints2, p=0.99, threshold=6.0, e=0.6):
    # assert keypoints1.shape[0] == keypoints2.shape[0]
    s = 4
    N = np.ceil(np.log(1 - p) / np.log(1 - (1 - e)**s))

    max_inliers_idx = []

    for _ in range(int(N)):
        idx = np.random.choice(range(len(matches)), size=s, replace=False)
        # print(idx)
        H = compute_homography(matches[idx], keypoints1, keypoints2)
        # print(H)

        inliers_idx = []

        for i,j in matches:
            x1, y1 = keypoints1[i].pt
            x2, y2 = keypoints2[j].pt
            p1 = np.array([x1, y1, 1])
            p2 = np.array([x2, y2, 1])
            p1_transformed = H @ p1
            p1_transformed /= p1_transformed[2]
            d = np.linalg.norm(p2 - p1_transformed)

            if d < threshold:
                inliers_idx.append([i, j])

        if len(inliers_idx) > len(max_inliers_idx):
            # print(len(inliers_idx))
            max_inliers_idx = inliers_idx

    max_inliers_idx = np.array(max_inliers_idx)

    H = compute_homography(max_inliers_idx, keypoints1, keypoints2)
    return H, max_inliers_idx


def estimate_homography(src_img, des_img):
    keypoints1, descriptors1, keypoints2, descriptors2 = get_keypoints_discriptors(src_img, des_img)
    matches = find_matches(descriptors1, descriptors2)
    H, inliers = RANSAC_homography(matches, keypoints1, keypoints2)
    return H