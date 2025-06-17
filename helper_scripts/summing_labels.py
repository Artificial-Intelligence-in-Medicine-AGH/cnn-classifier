import os
import sys

import numpy as np

# Hack: to make it run from `helper_scripts` subfolder
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from config import config

labelsFilePath = config.labels_file_path
labelsTable = np.genfromtxt(labelsFilePath, delimiter=',', dtype=str, usecols=(0,1), skip_header=1)

classes = { }

for i in range(labelsTable.shape[0]):
    classes_for_record = labelsTable[i,1].split('|')
    for key in classes_for_record:
        if not key in classes:
            classes[key] = 0
        classes[key] += 1

summing = sum(classes.values())

for key in classes.keys():
    print(f"Key: {key} | percentage: {round(classes[key]/summing * 100, 2)}%")

print("Total keys found: ", len(classes))


