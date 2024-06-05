# Scene Classification from Image

In this project, I implemented a scene classification system using various image processing and machine learning techniques. The methods implemented include:
- Tiny Image Repressntation with K-Nearest Neighbors(KNN) Classifier,
- Bag of Words (BoW) features with KNN Classifier, and
- BoW with Support Vector Machines (SVM) as Classifier.

### Problem Description

The goal of this project is to classify images into different scene categories. The dataset consists of images from various scene categories, such as forests, mountains, beaches, cities, and more. Each image is labeled with its corresponding scene category. The classification system aims to accurately predict the scene category of an unseen image based on its visual features.

### Overview of Methods

**Tiny Image Representation with KNN Classifier:** This method uses downsampled images for feature extraction and classifies them using a K-Nearest Neighbors classifier.

**Bag of Words (BoW) with KNN Classifier:** In this method, dense SIFT features are extracted, a visual dictionary is built, and then classification is done using K-Nearest Neighbors.

**BoW with Support Vector Machines (SVM):** Similar to the BoW with KNN, dense SIFT features are used to create a visual dictionary, but the classification is performed using a Support Vector Machine.


### Results

Below are the results obtained from the different methods:
<table>
  <tr>
    <th>Tiny Image with KNN</th>
    <th>BoW with KNN</th>
    <th>BoW with SVM</th>
  </tr>
  <tr>
    <td><img src="https://github.com/mohitydv09/scene-classification-using-Bow/assets/101336175/976ea1d2-a8d6-4d79-8408-cc76ff6ff6a0" width="300"/></td>
    <td><img src="https://github.com/mohitydv09/scene-classification-using-Bow/assets/101336175/31a6de03-daf7-4fc4-b6ac-5c8028043d8a" width="300"/></td>
    <td><img src="https://github.com/mohitydv09/scene-classification-using-Bow/assets/101336175/20d0e5c5-224b-4d6e-8145-1ea1839f42b0" width="300"></td>
  </tr>
</table>

