from autodistill_grounded_sam import GroundedSAM
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
import supervision as sv
import os
import cv2
import numpy as np

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# Classes
# nc: 17  # number of classes
# names: ['person', 'car', 'truck', 'cell phone', 'excavator', # 0-4 
#         'loader', 'crane', 'cone', 'hook', 'shovel', # 5-9
#          'payload', 'rigger', 'windbreaker', 'helmet', 'bar', # 10-14
#         'rope', 'barrier']  # 15 - 16
# base_model = GroundingDINO(ontology=CaptionOntology({"person": "person", "car": "car", "truck": "truck", "cell phone": "cell phone", "crawler crane": "crane", "cone": "cone", "helmet": "helmet"}))
# 15 classes
ontology_dict = {"person": "person", "car": "car", "truck": "truck", "cell phone": "cell phone", "Traffic cone": "cone", "helmet": "helmet", "machinery vehicle": "machinery vehicle"}

# label all images in a folder called `context_images`
base_model.label(
    # input_folder="/mnt/data/sibo/cone_dataset/chile43/train/images",
    # output_folder="/mnt/data/sibo/cone_dataset/chile43/train_dino"
    # input_folder="/mnt/data/sibo/chile_remote_videos/data_annotation/829_annotation/train/images",
    # output_folder="/mnt/data/sibo/chile_remote_videos/data_annotation/829_annotation/train_dino"
    # input_folder="/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/train/images",
    # output_folder="/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/train_dino"
    # input_folder="/mnt/data/sibo/GP45/20230517-0523/images",
    # output_folder="/mnt/data/sibo/GP45/20230517-0523/dino"
    # input_folder="/mnt/data/sibo/GP45/20230606-0612/images",
    # output_folder="/mnt/data/sibo/GP45/20230606-0612/dino"
    input_folder="/mnt/data/sibo/china_crane/shendong/images",
    output_folder="/mnt/data/sibo/china_crane/shendong/dino"
)