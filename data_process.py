import os
import glob
import numpy as np
import cv2

def read_all_img_pth(dir):
    """ Read all images from the folder
    
    Arguments:
        path {string} -- folder contains images
    
    Returns:
        string list -- all images path in the folder
    """
    img_path_list = glob.glob(os.path.join(dir, '*.jpg'))
    img_path_list.sort()
    return img_path_list


def save_res(path, idx, res):
    """ save results to a certain file

    Arguments:
        path -- file to save results
        idx -- Queries index, start from 1
        res -- results
    """    
    res = np.transpose(res)
    with open(path, 'a') as f:
        f.write(f'Q{idx}: ')
        np.savetxt(f, res[0, :], fmt='%d', delimiter=' ', newline=' ')
        f.write('\n')


def list_top_10(idx, line):
    """ List top 10 matches as required
    
    Arguments:
        idx -- start from 1
        line -- line from rankList file
    """    
    line = line.split()
    img_list = [int(x) for x in line[1:]][:10]
    img_list = [format(x, '05d') + '.jpg' for x in img_list]
    print(f'top 10: {img_list}')

    imstack = cv2.imread(f'./pg_data/pg_data/Images/{img_list[0]}')
    imstack = cv2.resize(imstack, (200, 200))
    imstack2 = cv2.imread(f'./pg_data/pg_data/Images/{img_list[5]}')
    imstack2 = cv2.resize(imstack2, (200, 200))
    img_list_half = img_list[:4]
    for img_pth in img_list_half:
        # print('./pg_data/pg_data/Images/' + img_pth)
        img = cv2.imread('./pg_data/pg_data/Images/' + img_pth)
        img = cv2.resize(img, (200, 200))
        imstack = np.hstack((imstack, img))
    img_list_half = img_list[6:]
    for img_pth in img_list_half:
        img = cv2.imread('./pg_data/pg_data/Images/' + img_pth)
        img = cv2.resize(img, (200, 200))
        imstack2 = np.hstack((imstack2, img))
    imstack = np.vstack((imstack, imstack2))
    cv2.imshow('stack', imstack)
    cv2.imwrite(f'./top_10_imgs/query{idx}_vgg.jpg', imstack)
    cv2.waitKey(0)


def img_resize(idx, pth):
    img = cv2.imread(pth)
    img = cv2.resize(img, (200, 200))
    cv2.imwrite(f'./top_10_imgs/q{idx}.jpg', img)


def main():
    with open('./rankList_vgg.txt', 'r') as f:
        rank_res = f.readlines()
        for idx, line in enumerate(rank_res):
            # we only care about query 1-5
            if idx == 5:
                break
            list_top_10(idx+1, line)


if __name__ == '__main__':
    main()
