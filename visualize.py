import os
import cv2
import numpy as np
import supervision as sv

# mode = "train"
mode = "valid"
#15 classes
base_path = "/mnt/data/sibo/GP45/202308/dino_7class/"
# base_path = "/mnt/data/sibo/GP45/20230517-0523/dino/"
# base_path = "/mnt/data/sibo/GP45/20230606-0612/dino_2class/"
# base_path = "/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/dino/"
# base_path = "/mnt/data/sibo/chile_remote_videos/Chile_713_data.v13i.yolov5pytorch/train_dino/"
# base_path = "/mnt/data/sibo/cone_dataset/chile43/train_dino/"
# base_path = "/mnt/data/sibo/cone_dataset/ArtificialTrafficCones.v1-basic-noaug.yolov5pytorch/"
# base_path = "/mnt/data/sibo/china_crane/north/dino_machinery/"
# base_path = "/mnt/data/sibo/china_crane/south/dino_machinery/"
# base_path = "/mnt/data/sibo/china_crane/shendong/dino_4class/"

ANNOTATIONS_DIRECTORY_PATH = os.path.join(base_path, mode, "labels")
IMAGES_DIRECTORY_PATH = os.path.join(base_path, mode, "images")
DATA_YAML_PATH = os.path.join(base_path, "data.yaml")
write_path = os.path.join(base_path, mode, "visualize")

# Check if the directory exists, if not, create it
if not os.path.exists(write_path):
    os.mkdir(write_path)
    
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH)

len(dataset)

image_names = list(dataset.images.keys())
# print(image_names)
mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

images = []
for image_name in image_names:
    image = dataset.images[image_name]
    annotations = dataset.annotations[image_name]
    labels = [
        dataset.classes[class_id]
        for class_id
        in annotations.class_id]
    annotates_image = mask_annotator.annotate(
        scene=image.copy(),
        detections=annotations)
    annotates_image = box_annotator.annotate(
        scene=annotates_image,
        detections=annotations,
        labels=labels)
    images.append(annotates_image)

# opencv write images to disk
print("Writing images in the folder:", write_path)
for i in range(len(images)):
    label_path = os.path.join(ANNOTATIONS_DIRECTORY_PATH, os.path.splitext(image_names[i])[0] + ".txt")
    
    if not os.path.exists(label_path):
        # print(f"Label for {image_names[i]} does not exist. Skipping this image.")
        continue

    print("Writing image:", image_names[i])
    cv2.imwrite(os.path.join(write_path, image_names[i]), images[i])


