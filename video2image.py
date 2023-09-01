import supervision as sv
from tqdm.notebook import tqdm
import os
# base_path = "/mnt/data/sibo/GP45/20230517-0523/"
# base_path = "/mnt/data/sibo/GP45/20230524-0605/"
base_path = "/mnt/data/sibo/GP45/20230606-0612/"

VIDEO_DIR_PATH = base_path + "videos"
IMAGE_DIR_PATH = base_path + "images"

if not os.path.exists(IMAGE_DIR_PATH):
    os.makedirs(IMAGE_DIR_PATH)

# pick every xth frame
# video is 23 - 30 fps
FRAME_STRIDE = 300

video_paths = sv.list_files_with_extensions(
    directory=VIDEO_DIR_PATH,
    extensions=["mov", "mp4"])

TEST_VIDEO_PATHS, TRAIN_VIDEO_PATHS = video_paths[:2], video_paths[2:]

for video_path in tqdm(TRAIN_VIDEO_PATHS):
    video_name = video_path.stem
    image_name_pattern = video_name + "-{:03d}.jpg"
    with sv.ImageSink(target_dir_path=IMAGE_DIR_PATH, image_name_pattern=image_name_pattern) as sink:
        for image in sv.get_video_frames_generator(source_path=str(video_path), stride=FRAME_STRIDE):
            sink.save_image(image=image)