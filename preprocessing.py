import os.path
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import config
import shutil
from PIL import Image
import torch


if len(sys.argv) < 2:
    print("ERROR: please provide a source directory as a argument of the script")
    exit()


REF_CLASSES = config.classes

if not None:
    REF_CLASSES.append("No Finding")

def get_ratios(labelsTable: np.ndarray, start: int, stop: int) -> np.ndarray:
    classes = {}

    for i in range(start, stop):
        classes_for_record = labelsTable[i, 1].split('|')
        for key in classes_for_record:
            if key not in classes:
                classes[key] = 0
            classes[key] += 1

    assert len(classes) == len(REF_CLASSES)

    classes = {key: classes[key] for key in REF_CLASSES}

    summing = sum(classes.values())

    percentage_arr = np.array(list(classes.values()), dtype=float) * 100.0 / summing

    return percentage_arr


labelsFilePath = config.labels_file_path
labelsTable = np.genfromtxt(labelsFilePath, delimiter=',', dtype=str, usecols=(0, 1), skip_header=1)

total = len(labelsTable)

IDX_70 = int(0.7*total)
IDX_90 = int(0.9*total)

while True:
    np.random.shuffle(labelsTable)

    try:
        ALL_RATIOS = get_ratios(labelsTable, 0, total)
        TRAIN_RATIOS = get_ratios(labelsTable, 0, IDX_70)
        VAL_RATIOS = get_ratios(labelsTable, IDX_70, IDX_90)
        TEST_RATIOS = get_ratios(labelsTable, IDX_90, total)
    except AssertionError:
        pass
    else:
        break


# CREATING LOGS PART
EVERYTHING_MATRIX = np.array([
    ALL_RATIOS,
    TRAIN_RATIOS,
    VAL_RATIOS,
    TEST_RATIOS,
])


datasets = ["All", "Train", "Val", "Test"]

ratios = { REF_CLASSES[i]: EVERYTHING_MATRIX[:, i] for i in range(len(REF_CLASSES))}

x = np.arange(len(datasets))
barW = 0.055
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in ratios.items():
    offset = barW * multiplier
    rects = ax.bar(x + offset, measurement, barW, label=attribute)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Ratio %')
ax.set_title('Ratios')
ax.set_xticks(x + barW, datasets)
# ax.legend(loc='upper right', ncols=3)
ax.set_ylim(0, 50)

folder_dir = os.path.join(config.logs_path, "preprocessing")
os.makedirs(folder_dir, exist_ok=True)
plt.savefig(os.path.join(folder_dir, "ratios.png"))

print(f"Ratios saved in {folder_dir}")


# RESIZE
SOURCE_DIR = sys.argv[1]
TEMP_DIR = os.path.join(SOURCE_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def find_image_path(filename:str):
    for root, _, files in os.walk(SOURCE_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None

print()  # Fixes display bugs with tqdm

for filename in tqdm(labelsTable[:,0], desc="Resizing"):
    src_path = find_image_path(filename)
    if src_path:
        try:
            img = Image.open(src_path)
            img = img.resize((config.final_img_width, config.final_img_width))
            img = np.array(img)

            # Make sure only 1 channel (grayscale)
            if len(img.shape) > 2 and img.shape[2] > 1:
                img = img[:, :, 0]

            dest_img_path = os.path.join(TEMP_DIR, filename)
            Image.fromarray(img).save(dest_img_path)
        except Exception as e:
            print(f"Issue with {filename}: {e}")
    else:
        print(f"Not found: {filename}")




# SPLITING
DESTINATION = config.dataset_path

def split_data(cases:np.ndarray ,subdir:str):
    print()
    os.makedirs(os.path.join(DESTINATION, subdir), exist_ok=True)
    for filename in tqdm(cases, desc=f"Splitting into {subdir}"):
        shutil.copy(os.path.join(TEMP_DIR, filename), os.path.join(DESTINATION, subdir, filename))


TRAIN_CASES = labelsTable[:IDX_70, 0]
VAL_CASES = labelsTable[IDX_70:IDX_90, 0]
TEST_CASES = labelsTable[IDX_90:, 0]

split_data(TRAIN_CASES, 'train')
split_data(VAL_CASES, 'val')
split_data(TEST_CASES, 'test')

shutil.rmtree(TEMP_DIR)

print(f"\ndata splitted into {DESTINATION}")
