import os
import cv2


def generate_dataset_txt(path='F:/NWPU-RESISC45/'):
    image_list = [[path + d + '/' + i for i in os.listdir(path + d)] for d in os.listdir(path)]
    class_num, image_num = len(image_list), len(image_list[0])
    with open('data.txt', 'a', encoding='utf-8') as f:
        for i in range(image_num):
            for j in range(class_num):
                f.write(image_list[j][i] + " " + str(j) + "\n")


def get_image_list(path, s):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.replace('\n', '').split(" ") for line in f.readlines()]
    return lines[int(s[0] * len(lines)): int(s[1] * len(lines))]


def get_image_for_path(path):
    return cv2.imread(path)

