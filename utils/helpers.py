import os
import cv2
import torch
import numpy as np
import torchvision
from PIL import Image
from skimage.exposure import match_histograms

def histogram_matching(source, reference):
    if len(source.shape) == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
    if len(reference.shape) == 2:
        reference = cv2.cvtColor(reference, cv2.COLOR_GRAY2RGB)
    matched_image = match_histograms(source, reference, multichannel=True)
    
    return source, reference, matched_image

def mean_squared_error(image1: np.ndarray, image2: np.ndarray) -> float:
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape.")
    return np.mean((image1 - image2) ** 2)


def detect_keypoints_and_match(image1, image2, detector, matcher, k=2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    keypoints1, descriptors1 = detector.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(gray2, None)
    
    matches = matcher.knnMatch(descriptors1, descriptors2, k=k)
    
    good_matches = []
    if k >= 2:
        for m in matches:
            if len(m) >= 2 and m[0].distance < 0.75 * m[1].distance:
                good_matches.append(m[0])
    else:
        for m in matches:
            good_matches.append(m[0])
    
    return keypoints1, keypoints2, good_matches

def compute_rotation_angle(keypoints1, keypoints2, matches):
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    affine_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, maxIters=3000)
    angle = np.arctan2(affine_matrix[0, 1], affine_matrix[0, 0])
    angle_degrees = np.degrees(angle)
    return angle_degrees

def refine_rotation_angle(similarity_model, circle1, circle2, estimated_angle, angle_check):
    best_angle = estimated_angle
    best_score = similarity_model(circle1, circle2.rotate(estimated_angle))
    for angle in np.arange(estimated_angle - 180, estimated_angle + 180, angle_check):
        score = similarity_model(circle1, circle2.rotate(angle))
        if best_score < score:
            best_angle = angle
            best_score = score
    return best_angle, best_score

def reverse_rotation(image1, image2, detector, matcher):
    keypoints1, keypoints2, good_matches = detect_keypoints_and_match(image1, image2, detector, matcher, k=10)
    if len(good_matches) < 3: raise ValueError("Not enough matches are found.")
    
    return compute_rotation_angle(keypoints1, keypoints2, good_matches)


def get_aligning_angle(similarity_model, circle1, circle2, angle_check):
    resizer = torchvision.transforms.Resize([512, 512])
    resized_image1 = np.asarray(resizer.forward(circle1))
    resized_image2 = np.asarray(resizer.forward(circle2))

    gray_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)

    return refine_rotation_angle(similarity_model, Image.fromarray(gray_image1), Image.fromarray(gray_image2), 0, angle_check)[0]

def create_sum_map(map_shape, overlap_px, patches):
    sum_map = np.zeros(map_shape)
    patch_size = patches[0].shape[0]

    patch_counter = 0
    for i in range((map_shape[0]  - patch_size) // overlap_px + 1): # rows
        for j in range((map_shape[1]  - patch_size) // overlap_px + 1): # columns
            sum_map[i * overlap_px: patch_size + i * overlap_px, j * overlap_px: patch_size + j * overlap_px] += patches[patch_counter]
            patch_counter += 1
        
    return sum_map

def perform_overlap_bilinear(sum_map, overlap_px):
    weight_map = np.ones_like(sum_map)
    
    weight_map[overlap_px:-overlap_px, overlap_px:-overlap_px] = 4  # center part
    weight_map[overlap_px:-overlap_px, :overlap_px] = np.linspace(2, 4, overlap_px)  # left part
    weight_map[overlap_px:-overlap_px, -overlap_px:] = np.linspace(4, 2, overlap_px)  # right part
    weight_map[:overlap_px, overlap_px:-overlap_px] = np.linspace(2, 4, overlap_px)[:, np.newaxis]  # upper part
    weight_map[-overlap_px:, overlap_px:-overlap_px] = np.linspace(4, 2, overlap_px)[:, np.newaxis]  # lower part
    
    for i in range(overlap_px):
        for j in range(overlap_px):
            weight_map[i, j] = max(2, weight_map[i, j])  # top-left
            weight_map[i, -j-1] = max(2, weight_map[i, -j-1])  # top-right
            weight_map[-i-1, j] = max(2, weight_map[-i-1, j])  # bottom-left
            weight_map[-i-1, -j-1] = max(2, weight_map[-i-1, -j-1])  # bottom-right
    
    return sum_map / weight_map

def overlap_patches(map_shape, overlap_px, patches):
    sum_map = create_sum_map(map_shape, overlap_px, patches)
    # return perform_overlap_mean(sum_map, overlap_px)
    return perform_overlap_bilinear(sum_map, overlap_px)

# def crop_image(image: np.array, path_size=500):
#     patches = []
#     for h_step in range(image.shape[0] // path_size):
#         for w_step in range(image.shape[1] // path_size):
#             start_h = h_step * path_size
#             start_w = w_step * path_size
#             patch = image[start_h: start_h + path_size, start_w: start_w + path_size, ...]
#             patches.append(patch)
#     return patches


def crop_image(image: np.ndarray, patch_size: int = 500, stride: int = 500) -> list[np.ndarray]:    
    patches = []
    
    for i in range((image.shape[0]  - patch_size) // stride + 1): # rows
        for j in range((image.shape[1]  - patch_size) // stride + 1): # columns
            current_patch = image[i * stride: patch_size + i * stride, j * stride: patch_size + j * stride]
            patches.append(current_patch)
            
    return patches


def overlap_and_crop(patches: list[np.ndarray]) -> np.ndarray:
    overlapped_patches = overlap_patches([2000, 2000], 256, patches)
    return crop_image(overlapped_patches, 500, 500)