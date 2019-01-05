import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import scipy
from scipy import spatial
import itertools

IMAGE_DIR = os.getcwd() + '\\test'  # =========== change this directory to your own folder ===============
os.chdir(IMAGE_DIR)

image_files = os.listdir(os.getcwd())
# print(len(image_files))
# print(image_files[0])
# print(cv2.imread(image_files[0]).shape)


def filter_images(images):
    image_list = []
    for image in images:
        try:
            assert cv2.imread(image).shape[2] == 3, "gray image check mark"
            image_list.append(image)
        except AssertionError as e:
            print(e)
    return image_list


def img_gray(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    return image


# get columns and rows as list
def resize(image, width=30, height=30):
    row_res = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).flatten()
    col_res = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA).flatten('F')
    return row_res, col_res


# gradient / see if values increase or decrease as a trend
def intensity_diff(row_res, col_res):
    difference_row = np.diff(row_res)  # diff >> n - (n-1)
    difference_col = np.diff(col_res)
    difference_row = difference_row > 0  # list of booleans now
    difference_col = difference_col > 0
    return np.vstack((difference_row, difference_col)).flatten()  # same thing as np.hstack


def difference_score(image, width=30, height=30):
    gray = img_gray(image)
    row_res, col_res = resize(gray, width, height)
    difference = intensity_diff(row_res, col_res)
    return difference  # returns boolean list


def hamming_distance(image1, image2):
    score = scipy.spatial.distance.hamming(image1, image2)
    return score


def difference_score_hamming(image_list):
    ds_dict = {}
    duplicates = []
    for image in image_list:
        ds = tuple(difference_score(image))
        print(type(ds))
        if ds not in ds_dict:
            print("poop")
            ds_dict[image] = ds
        else:
            duplicates.append((image, ds_dict[image]))
    return duplicates, ds_dict


# main
image_files = filter_images(image_files)
duplicates_out, ds_dict_out = difference_score_hamming(image_files)
print(len(duplicates_out))
print(len(ds_dict_out.keys()))

# compare all combinations of images for similarity
for k1, k2 in itertools.combinations(ds_dict_out, 2):  # non repeating combinations of 2
    if hamming_distance(ds_dict_out[k1], ds_dict_out[k2]) < .07:  # =========== high decimal = sensitive ===============
        duplicates_out.append((k1, k2))

print(len(duplicates_out))

for file_names in duplicates_out:
    try:
        plt.subplot(121), plt.imshow(cv2.cvtColor(cv2.imread(file_names[0]), cv2.COLOR_BGR2RGB))
        plt.title('Duplicate'), plt.xticks([]), plt.yticks([])

        plt.subplot(122), plt.imshow(cv2.cvtColor(cv2.imread(file_names[1]), cv2.COLOR_BGR2RGB))
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.show()

    except OSError as e:
        continue

