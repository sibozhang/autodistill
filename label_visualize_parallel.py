import supervision as sv
import os
import cv2
import sys
import concurrent.futures

from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

# Ensure directories exist
def ensure_directories_exist(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def annotate_and_save_image(image_name, image, dataset, mask_annotator, box_annotator, WRITE_PATH, gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Set GPU ID for this thread
    annotations = dataset.annotations[image_name]
    labels = [dataset.classes[class_id] for class_id in annotations.class_id]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=annotations)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=annotations, labels=labels)
    cv2.imwrite(os.path.join(WRITE_PATH, image_name), annotated_image)

def process_images(mode, num_gpus=8):
    ANNOTATIONS_DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER, mode, 'labels')
    IMAGES_DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER, mode, 'images')
    WRITE_PATH = os.path.join(OUTPUT_FOLDER, mode, 'visualize')
    
    base_model.label(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER)
    dataset = sv.DetectionDataset.from_yolo(images_directory_path=IMAGES_DIRECTORY_PATH, annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH, data_yaml_path=DATA_YAML_PATH)
    
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(annotate_and_save_image, image_name, image, dataset, mask_annotator, box_annotator, WRITE_PATH, i % num_gpus): image_name
                   for i, (image_name, image) in enumerate(dataset.images.items())}
        
        for future in concurrent.futures.as_completed(futures):
            image_name = futures[future]
            try:
                future.result()
                print(f"Successfully processed image: {image_name}")
            except Exception as exc:
                print(f"An error occurred while processing image {image_name}: {exc}")

# Define constants
BASE_PATH = "/mnt/data/sibo/GP45/202402/helmet/"
INPUT_FOLDER = os.path.join(BASE_PATH, 'images/')
OUTPUT_FOLDER = os.path.join(BASE_PATH, 'dino_7class/')
DATA_YAML_PATH = os.path.join(OUTPUT_FOLDER, 'data.yaml')
ontology_dict = {"person": "person", "car": "car", "truck": "truck", "Traffic cone": "cone", "helmet": "helmet", "machinery vehicle": "machinery vehicle", "hook": "hook"}
base_model = GroundingDINO(ontology=CaptionOntology(ontology_dict))

ensure_directories_exist([os.path.join(OUTPUT_FOLDER, mode, 'labels') for mode in ['train', 'val']] + 
                         [os.path.join(OUTPUT_FOLDER, mode, 'images') for mode in ['train', 'val']] +
                         [os.path.join(OUTPUT_FOLDER, mode, 'visualize') for mode in ['train', 'val']])

process_images('train')
process_images('val')
