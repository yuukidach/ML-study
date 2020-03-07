import numpy as np
from numpy import linalg as LA
from keras.preprocessing import image
from keras.applications.vgg16 import (preprocess_input,
                                      VGG16)
import h5py
import os

from data_process import (read_all_img_pth,
                          save_res)

# use the trained VGG
class VGGNet:
    def __init__(self):
        # weights: 'imagenet'
        # pooling: 'max' or 'avg'
        # input_shape: (width, height, 3), width and height should >= 48
        self.input_shape = (224, 224, 3)
        self.weight = 'imagenet'
        self.pooling = 'max'
        self.model = VGG16(weights = self.weight, input_shape = (self.input_shape[0], self.input_shape[1], self.input_shape[2]), pooling = self.pooling, include_top = False)
        self.model.predict(np.zeros((1, 224, 224 , 3)))

    def extract_feat(self, img_path):
        """ use vgg16 to extract features
        
        Arguments:
            img_path
        
        Returns:
            nomalized feature vector
        """        
        img = image.load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = self.model.predict(img)
        norm_feat = feat[0]/LA.norm(feat[0])
        return norm_feat


vgg = VGGNet()
names = []
feats = []

# last time when I ran sift, I forgot to create a database
# OMG!!! TAT
def build_database():
    """ create database for images
    """
    feats = []
    img_list = read_all_img_pth('./pg_data/pg_data/Images')

    for idx, img_pth in enumerate(img_list):
        print(f'Computing image {idx+1}...')
        name = img_pth.split('/')[-1]
        norm_feat = vgg.extract_feat(img_pth)
        feats.append(norm_feat)
        names.append(name.encode())
    feats = np.array(feats)
    print('Now saving the database...')
    h5f = h5py.File('img_feat_vgg.h5', 'w')
    h5f.create_dataset('dataset1', data=names)
    h5f.create_dataset('dataset2', data=feats)
    h5f.close()


def read_database():
    """ read data from database we create
    """    
    global feats
    h5f = h5py.File('img_feat_vgg.h5', 'r')
    names = h5f['dataset1'][:]
    feats = h5f['dataset2'][:]
    h5f.close()


def vgg_score(q_idx, img_pth):
    """ use vgg to score the images
    
    Arguments:
        q_idx -- start from 1
        img_pth -- path of query images
    
    Returns:
        [type] -- [description]
    """    
    query_feat = vgg.extract_feat(img_pth)
    scores = np.dot(query_feat, feats.T)
    rank = np.argsort(scores)[::-1]

    return rank + 1


def main():
    query_list = read_all_img_pth('./pg_data/pg_data/Queries')
    read_database()
    
    for idx, img_pth in enumerate(query_list):
        print(f'Processing query {idx+1}')
        res = vgg_score(idx + 1, img_pth)
        res = np.array([res] * 2)
        res = res.T
        save_res('rankList_vgg.txt', idx + 1, res)
    

if __name__ == '__main__':
    if not os.path.exists('img_feat_vgg.h5'):
        build_database()
    main()
