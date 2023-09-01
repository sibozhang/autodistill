import os
import cv2
import numpy as np
import supervision as sv
import shutil

# specify mode
# mode = "valid"
mode = "train"

# specify directories
base_path = "/mnt/data/sibo/GP45/202308/dino_7class"
# base_path = "/mnt/data/sibo/GP45/20230517-0523/dino_2class"
# base_path = "/mnt/data/sibo/GP45/20230606-0612/dino_2class"
# base_path = "/mnt/data/sibo/china_crane/north/dino_machinery/"
# base_path = "/mnt/data/sibo/china_crane/south/dino_machinery/"
# yolo_label_dir = os.path.join('/mnt/scratch/sibo/yolov5/runs/detect/20230606-0612_valid', 'labels')
# yolo_label_dir = os.path.join('/mnt/scratch/sibo/yolov5/runs/detect/20230606-0612_train', 'labels')
# yolo_label_dir = os.path.join('/mnt/scratch/sibo/yolov5/runs/detect/china_crane_north', mode, 'labels')
# yolo_label_dir = os.path.join('/mnt/scratch/sibo/yolov5/runs/detect/china_crane_south', mode, 'labels')
# yolo_label_dir = os.path.join('/mnt/scratch/sibo/yolov5/runs/detect/202308_train', 'labels')
yolo_label_dir = os.path.join('/mnt/scratch/sibo/yolov5/runs/detect/202308', mode, 'labels')
autodistill_label_dir = os.path.join(base_path, mode, 'labels_7class')
output_dir = os.path.join(base_path, mode, 'labels')

ANNOTATIONS_DIRECTORY_PATH = output_dir
IMAGES_DIRECTORY_PATH = os.path.join(base_path, mode, 'images')
DATA_YAML_PATH = os.path.join(base_path, 'coco_chile_dino_15class.yaml')
# DATA_YAML_PATH = os.path.join(base_path, 'coco_machine_81class.yaml') #81class
write_path = os.path.join(base_path, mode, 'visualize')


# list of directories to check
dirs_to_check = [yolo_label_dir, autodistill_label_dir, IMAGES_DIRECTORY_PATH, DATA_YAML_PATH]

# check if directories exist, if not, exit
for dir_path in dirs_to_check:
    if not os.path.exists(dir_path):
        print(f"Error: The directory {dir_path} does not exist.")
        exit()

# list of directories to create if they don't exist
dirs_to_create = [output_dir, write_path]

# check if directories exist, create if necessary
for dir_path in dirs_to_create:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
print ("Directories checked and created if necessary.")

# for chile
# yolo classes:
# nc: 15  # number of classes
# names: ['person', 'car', 'truck', 'cell phone', 'excavator', # 0-4 
#         'loader', 'crane', 'cone', 'hook', 'shovel', # 5-9
#          'payload', 'helmet', 'bar', 'rope', 'barrier'] # 10-14
# dino class names:
# - person
# - car
# - truck
# - cell phone
# - cone
# - helmet
# - machinery vehicle
# nc: 7
# # specify the class to replace
# yolo_class_id_to_remove = 7
# cone:
# autodistill_class_id = 4 
# new_class_id = 7 
# helmet:
autodistill_class_ids = [0, 4, 5]
new_class_ids = [0, 7, 11]          
#person
# autodistill_class_id = 0
# new_class_id = 0

# for shendong
# specify the class to add
# yolo_class_id_to_remove = 7
# autodistill_class_id = 3
# new_class_id = 80

# iterate over files in yolo_label_dir
for filename in os.listdir(yolo_label_dir):
    if filename.endswith(".txt"):
        yolo_file_path = os.path.join(yolo_label_dir, filename)
        autodistill_file_path = os.path.join(autodistill_label_dir, filename)
        output_file_path = os.path.join(output_dir, filename)

        try:
            # read yolo labels
            with open(yolo_file_path, 'r') as yolo_file:
                yolo_labels = yolo_file.readlines()
            
            modified_labels = [line for line in yolo_labels]

            # check if the same label file exists in autodistill_label_dir
            if os.path.isfile(autodistill_file_path):
                with open(autodistill_file_path, 'r') as autodistill_file:
                    autodistill_labels = autodistill_file.readlines()
                
                # Loop through each autodistill class ID
                for autodistill_class_id, new_class_id in zip(autodistill_class_ids, new_class_ids):
                    # add lines start with autodistill_class_id from autodistill_label_dir 
                    for autodistill_label in autodistill_labels:
                        autodistill_class_id_str, _, rest_of_label = autodistill_label.partition(' ')
                        if autodistill_class_id_str == str(autodistill_class_id):
                            new_label = str(new_class_id) + ' ' + rest_of_label
                            modified_labels.append(new_label)

            # write the modified labels to output file
            with open(output_file_path, 'w') as output_file:
                output_file.writelines(modified_labels)

            print(f'Successfully processed {filename}')
        except Exception as e:
            print(f'Error processing {filename}: {str(e)}')

# iterate over files in autodistill_label_dir
for filename in os.listdir(autodistill_label_dir):
    if filename.endswith(".txt"):
        autodistill_file_path = os.path.join(autodistill_label_dir, filename)
        yolo_file_path = os.path.join(yolo_label_dir, filename)

        # if label file in autodistill_label_dir but not in yolo_label_dir, copy to there
        if not os.path.isfile(yolo_file_path):
            shutil.copy2(autodistill_file_path, yolo_label_dir)

# visualize the labels
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=IMAGES_DIRECTORY_PATH,
    annotations_directory_path=ANNOTATIONS_DIRECTORY_PATH,
    data_yaml_path=DATA_YAML_PATH)

print(f"Total Images in Dataset: {len(dataset)}")

image_names = list(dataset.images.keys())
# print(f"Image Names: {image_names}")

mask_annotator = sv.MaskAnnotator()
box_annotator = sv.BoxAnnotator()

images = []
for image_name in image_names:
    try:
        image = dataset.images[image_name]
        annotations = dataset.annotations[image_name]
        # print(f"For image {image_name}, annotations: {annotations}")
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
    except Exception as e:
        print(f'Error processing image {image_name}: {str(e)}')

# opencv write images to disk
for i in range(len(images)):
    print("Writing image: ", image_names[i])
    try:
        cv2.imwrite(os.path.join(write_path, image_names[i]), images[i])
    except Exception as e:
        print(f'Error writing image {image_name}: {str(e)}')
