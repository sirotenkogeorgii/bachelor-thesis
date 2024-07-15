import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics.pairwise import cosine_similarity
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from .helpers import get_aligning_angle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageSimilarityModel:
    def __init__(self):
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
        self.model.classifier = nn.Identity().to(device)
        self.model.eval()
        self.cache = None

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _image_to_tensor(self, image):
        image = image.convert('RGB')
        image = self.preprocess(image).unsqueeze(0)
        return image.to(device)

    def _extract_features(self, image_tensor):
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.cpu().numpy().flatten()

    def __call__(self, image1, image2):
        image2_tensor = self._image_to_tensor(image2)
        features2 = self._extract_features(image2_tensor)

        if self.cache is None:
            self.cache = self._extract_features(self._image_to_tensor(image1))


        similarity = cosine_similarity([self.cache], [features2])
        return similarity[0][0]


class Preprocessor:
    def __init__(self, segmentation_weights_path):
        self.model = SegformerForSemanticSegmentation.from_pretrained(segmentation_weights_path).to(device)
        self.similarity_model = ImageSimilarityModel()

    def _get_circle(self, image, new_size=None):
        image_processor = SegformerImageProcessor(reduce_labels=True)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad(): outputs = self.model(pixel_values=pixel_values)
        predicted_segmentation_map = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
        numpy_image = np.array(image)
        mask = predicted_segmentation_map.astype(np.uint8)
        result = cv2.bitwise_and(numpy_image, numpy_image, mask=mask)
        coords = np.column_stack(np.where(mask))
        
        if coords.size != 0:
            top_left = coords.min(axis=0)
            bottom_right = coords.max(axis=0) + 1
            cropped_image = Image.fromarray(result).crop((*top_left[::-1], *bottom_right[::-1]))
        else: cropped_image = Image.fromarray(mask)

        if new_size:
            resizer = torchvision.transforms.Resize([new_size[0], new_size[1]])
            cropped_image = resizer.forward(cropped_image)
        return cropped_image
    
    def _filter_no_circles(self, circle_pairs): return [pair for pair in circle_pairs if np.sum(pair[0]) != 0 and np.sum(pair[1]) != 0]
    
    def _take_snaphot(self, video, sec):
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(sec * fps)
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        frame = video.read()[1]
        return frame

    def extract_circles(self, video_path1, video_path2, start_time=2, num_pairs=1, interval=10):
        cap1 = cv2.VideoCapture(video_path1)
        cap2 = cv2.VideoCapture(video_path2)
        if not cap1.isOpened(): raise Exception(f"Error: Could not open video file {video_path1}")
        if not cap2.isOpened(): raise Exception(f"Error: Could not open video file {video_path2}")
        
        frame_pairs = [(
            Image.fromarray(self._take_snaphot(cap1, start_time + i*interval)),
            Image.fromarray(self._take_snaphot(cap2, start_time + i*interval))
            )
                for i in range(num_pairs)]

        cap1.release()
        cap2.release()

        cutout_circles = [(self._get_circle(pair[0], [2000, 2000]), self._get_circle(pair[1], [2000, 2000])) for pair in frame_pairs]
        return self._filter_no_circles(cutout_circles)
    
    def normalize_to_start_pos(self, circle_pairs, interval): return [(pair[0].rotate(i*interval*(360 / 81)), pair[1].rotate(i*interval*(360 / 81))) for i, pair in enumerate(circle_pairs)] # it takes 81 seconds to complete 360 degrees

    def align_circles(self, circle_pairs, angle_check=2):
        if len(circle_pairs) == 0: return

        aligning_angle = get_aligning_angle(self.similarity_model, circle_pairs[0][0], circle_pairs[0][1], angle_check)
        return [(pair[0], pair[1].rotate(aligning_angle)) for pair in circle_pairs]
    
    


