import numpy as np
import cv2
from matplotlib import pyplot as plt
from data_process import (read_all_img_pth, 
                          save_res)
                         
FLANN_INDEX_KDTREE = 0
matcher = cv2.FlannBasedMatcher(dict(algorithm = FLANN_INDEX_KDTREE, trees = 5), {})
sift = cv2.xfeatures2d.SIFT_create(450)

img_feature = []

def get_feature(img_path):
    """ get key points and descriptor
    
    Arguments:
        img_path {string} -- relative path of the image 
    
    Returns:
        [key point, descriptor]
    """    
    # read in gray scale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(img, None)
    return [kp, des]


def sift_score(q_idx, src_path):
    """ score the similarity for images in images folder
    
    Arguments:
        q_idx {int} -- index of query image, start from 0
        src_path {string} -- relative path of the query image
    
    Returns:
        res -- [[img index, score], ...]
    """    
    res = []

    # read image in gray scale
    src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    # get the descriptor
    kp1, des1 = sift.detectAndCompute(src_img, None)
    kp1_num = len(kp1)

    for idx, [_, des2] in enumerate(img_feature):
        # match feature points
        matches = matcher.knnMatch(des1, des2, 2)
        # sort matched points
        matches = sorted(matches, key=lambda x: x[0].distance)
        # find the valuable points
        val_pt = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]
        res.append([idx + 1, len(val_pt) / kp1_num])

    res = np.array(res)
    return res


def main():
    query_list = read_all_img_pth('./pg_data/pg_data/Queries')
    img_list = read_all_img_pth('./pg_data/pg_data/Images')

    for idx, img_path in enumerate(img_list):
        print(f'Computing image {idx+1}...')
        img_feature.append(get_feature(img_path))

    for idx, img_path in enumerate(query_list):
        print(f'Processing query {idx+1}...')
        res = sift_score(idx, img_path)
        res = res[np.argsort(res[:, 1])[::-1]]
        save_res('rankList_sift.txt', idx + 1, res)


if __name__ == '__main__':
    main()
