# ASSIGNMENT 1

## 1. Goal

Implement 2 object matching methods that we learnt in the course to calculate the similarity among query images and target images. There are total 50 query images and 28493 target images.

## 2. Background

In this assignment I use SIFT and VGG-16 to get the feature extract features from images and calculate there distance to decide their similarity.

### 2.1. SIFT

The scale-invariant feature transform (SIFT) is a feature detection algorithm in computer vision to detect and describe  local feature in images. Its major stages are:

1. Scale-space extrema detection
2. Keypoint localization
3. Orientation assignment
4. Keypoint descriptor

### 2.2. VGG-16

VGG-16 is a convolutional neural network architecture, it’s name VGG-16 comes from the fact that it has 16 layers. It’s layers consists of Convolutional layers, Max Pooling layers, Activation layers, Fully connected layers.

![VGG-16 Layers](./markdown-res/vgg16-layer.png)

## 3. Process

### 3.1. Run With SIFT

### 3.1.1. Steps

I first run the test using SIFT. The steps are:

1. Extract features for all target images with SIFT
2. Get one query image and extract its features
3. Use kNN and K-D Trees to sort their Euclidean distance
4. If there is another query image, go back to step 2
5. Save the result in a text file

### 3.1.2. Results

Since the report cannot be longer than 5 pages, I top 10 matched images for query image 1-5 in `top_10_imgs/` folder. Please check that.

### 3.2 Run With VGG-16

### 3.2.1 Steps

Due to VGG-16 is based on CNN, but there is no enough training set for us to train our model. So, I downloaded the already trained model from the `keras` Python library and use it to extract features from images.

The major steps for running with VGG-16 are:

1. Download the already trained model
2. Extract features for all target images with VGG-16
3. Save features into a database
4. Get one query image and extract its features
5. Calculate cosine similarity for query image features and all target image features
6. If there is another query image, go back to step 4
7. Save the result in a text file

### 3.2.2. Results

For detailed results, please check `rankList_sift.txt` here, I put the top 10 matched result for query images 1-5.

Query 1

![top 10](./top_10_imgs/query1_vgg.jpg)

Query 2

![top 10](./top_10_imgs/query2_vgg.jpg)

Query 3

![top 10](./top_10_imgs/query3_vgg.jpg)

Query 4

![top 10](./top_10_imgs/query4_vgg.jpg)

Query 5

![top 10](./top_10_imgs/query5_vgg.jpg)

## 4. Conclusion

We can calculate the similarity between 2 images with their features. Comparing the 2 methods we use, VGG-16 is faster than SIFT when extracting features from images. And from the top 10 matched results for query 1-5, we can aslo know that using VGG-16 to do object matching is more precise than SIFT.
