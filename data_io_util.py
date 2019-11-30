import numpy as np

INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]

LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


def load_data(files):
    arr = []
    for file_name in files:
        with open(file_name, 'r') as f:
            rows = [row.replace('  ', ' ').strip().split(' ') for row in f]
            arr.append([np.array(ele, dtype=np.float32) for ele in rows])
    return np.transpose(np.array(arr), (1, 2, 0))
