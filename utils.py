import os
import numpy as np
from PIL import Image
from collections import defaultdict

train_path = '/home/enningxie/Downloads/Cell_/train'
test_path = '/home/enningxie/Downloads/Cell_/test'
train_label_path = '/home/enningxie/Downloads/Cell_/train.txt'
test_label_path = '/home/enningxie/Downloads/Cell_/test.txt'


# return x, y
def _load_data(data_path, label_path):
    data_label_dict = defaultdict(list)
    data_list = []
    label_list = []
    for file in os.listdir(data_path):
        # img_path
        tmp_path = os.path.join(data_path, file)
        # load img
        tmp_img = Image.open(tmp_path)
        # reshape img to flatten
        tmp_img_data = np.array(tmp_img.getdata()).reshape((tmp_img.size[0]*tmp_img.size[1]*3, )).astype(np.float32)
        # normalize_op
        tmp_img_data /= 255.
        data_label_dict[file].append(tmp_img_data)

    with open(label_path, 'r') as f:
        labels = f.readlines()
    for label in labels:
        label_value = label.split('\t')[1][0]
        label_key = label.split('\t')[0].split('\\')[-1]
        if label_key in data_label_dict.keys():
            data_label_dict[label_key].append(label_value)

    for value in data_label_dict.values():
        data_list.append(value[0])
        label_list.append(value[1])

    assert len(data_list) == len(label_list), 'process data error!'
    return data_list, label_list


# return train_x, train_y, test_x, test_y
def return_data(train_path, train_label_path, test_path, test_label_path):
    train_x, train_y = _load_data(train_path, train_label_path)
    test_x, test_y = _load_data(test_path, test_label_path)
    return train_x, train_y, test_x, test_y
