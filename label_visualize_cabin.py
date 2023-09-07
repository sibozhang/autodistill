import supervision as sv
import os
import cv2
import sys

# sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

def process_images(mode):
    # Constants depending on the mode
    ANNOTATIONS_DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER,  mode, 'labels')
    IMAGES_DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER,  mode, 'images')
    WRITE_PATH = os.path.join(OUTPUT_FOLDER,  mode, 'visualize')

    # Check and create directories
    directories = [ANNOTATIONS_DIRECTORY_PATH, IMAGES_DIRECTORY_PATH, WRITE_PATH]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Label all images in the folder
    base_model.label(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER)

    # Load dataset
    dataset = sv.DetectionDataset.from_yolo(images_directory_path=IMAGES_DIRECTORY_PATH, annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH, data_yaml_path=DATA_YAML_PATH)

    # Initialize annotators
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()

    # Annotate images and write them to disk
    for image_name, image in dataset.images.items():
        annotations = dataset.annotations[image_name]
        labels = [dataset.classes[class_id] for class_id in annotations.class_id]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=annotations)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations, labels=labels)
        print(f"Writing image: {image_name}")
        cv2.imwrite(os.path.join(WRITE_PATH, image_name), annotated_image)

# Define constants that are mode-independent
# (These remain outside the function)
BASE_PATH = "/mnt/data/sibo/GP45/202309/cabin"
INPUT_FOLDER = os.path.join(BASE_PATH, 'images/')
OUTPUT_FOLDER = os.path.join(BASE_PATH, 'dino_2class/')
DATA_YAML_PATH = os.path.join(OUTPUT_FOLDER, 'data.yaml')
ontology_dict = {"person": "person", "cell phone": "cell phone"}
base_model = GroundingDINO(ontology=CaptionOntology(ontology_dict))

# Run for both modes
process_images('train')
process_images('val')

# # specify mode
# mode = "valid"
# # mode = "train"

# # Define constants
# # BASE_PATH = "/mnt/data/sibo/GP45/20230517-0523/"
# # BASE_PATH = "/mnt/data/sibo/GP45/20230524-0605/"
# # BASE_PATH = "/mnt/data/sibo/GP45/20230606-0612/"
# BASE_PATH = "/mnt/data/sibo/GP45/202308/cabin"
# # BASE_PATH = "/mnt/data/sibo/china_crane/north/"
# # BASE_PATH = "/mnt/data/sibo/china_crane/south/"
# # BASE_PATH = "/mnt/data/sibo/china_crane/shendong/"

# INPUT_FOLDER = os.path.join(BASE_PATH, 'images/')
# OUTPUT_FOLDER = os.path.join(BASE_PATH, 'dino_2class/')
# # INPUT_FOLDER = os.path.join(BASE_PATH, 'images_0.25fps/')
# # OUTPUT_FOLDER = os.path.join(BASE_PATH, 'dino_4class/')
# ANNOTATIONS_DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER,  mode, 'labels')
# IMAGES_DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER,  mode, 'images')
# DATA_YAML_PATH = os.path.join(OUTPUT_FOLDER, 'data.yaml')
# WRITE_PATH = os.path.join(OUTPUT_FOLDER,  mode, 'visualize')

# # Check and create directories
# directories = [OUTPUT_FOLDER, ANNOTATIONS_DIRECTORY_PATH, IMAGES_DIRECTORY_PATH, WRITE_PATH]
# for directory in directories:
#     if not os.path.exists(directory):
#         os.makedirs(directory)

# # names: ['person', 'car', 'truck', 'cell phone', 'excavator', # 0-4 
# #         'loader', 'crane', 'cone', 'hook', 'shovel', # 5-9
# #          'payload', 'rigger', 'windbreaker', 'helmet', 'bar', # 10-14
# #         'rope', 'barrier']  # 15 - 16
# # Define ontology and initialize base model
# # cabin 2 class
# ontology_dict = {"person": "person", "cell phone": "cell phone"}
# # 7 classes
# # ontology_dict = {"person": "person", "car": "car", "truck": "truck", "cell phone": "cell phone", "Traffic cone": "cone", "helmet": "helmet", "machinery vehicle": "machinery vehicle"}
# # # 15 classes
# # ontology_dict = {"person": "person", "car": "car", "truck": "truck", "cell phone": "cell phone", "crawler crane": "crane", "cone": "cone", "hook": "hook", "shovel": "shovel", "payload": "payload", "helmet": "helmet", "bar": "bar", "rope": "rope", "barrier": "barrier"}
# # 4 class: shendong
# # ontology_dict = {"person": "person", "car": "car", "truck": "truck", "machinery vehicle": "machinery vehicle"}
# # 6 class
# # ontology_dict = {"person": "person", "car": "car", "truck": "truck", "cell phone": "cell phone", "cone": "cone", "helmet": "helmet"}

# base_model = GroundingDINO(ontology=CaptionOntology(ontology_dict))

# # Label all images in the folder
# base_model.label(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER)

# # Load dataset
# dataset = sv.DetectionDataset.from_yolo(images_directory_path=IMAGES_DIRECTORY_PATH, annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH, data_yaml_path=DATA_YAML_PATH)

# # Initialize annotators
# mask_annotator = sv.MaskAnnotator()
# box_annotator = sv.BoxAnnotator()

# # Annotate images and write them to disk
# for image_name, image in dataset.images.items():
#     annotations = dataset.annotations[image_name]
#     labels = [dataset.classes[class_id] for class_id in annotations.class_id]
#     annotated_image = mask_annotator.annotate(scene=image.copy(), detections=annotations)
#     annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations, labels=labels)
#     print(f"Writing image: {image_name}")
#     cv2.imwrite(os.path.join(WRITE_PATH, image_name), annotated_image)
