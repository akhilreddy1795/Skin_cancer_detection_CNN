# Data Science Skin Cancer detection using Convolution Neural Networks  based model: Project Overview
---
* I have used a pre trained model using **Transfer-learning** and all the model building is based on custom model
* Objective is to identify the image belongs to which class there are 9 classes in total
* used sgd as optimiser and evaluated on accuracy as metric
* Finding the class imbalance is present or not if present treating it with data Augmentation

## Code and Resources Used
---
* **Python Version:** 3.7
* **Packages:** pandas, numpy, seaborn, pathlib, tensorflow, PIL, os, tensorflow.keras

## Loading and visualising Data
---
* dataset consists of about **2357 images of skin cancer types,** The dataset contains 9 sub-directories in each train and test subdirectories. The 9 sub-directories contains the images of 9 skin cancer types respectively.
* Load using **Keras.preprocessing**
* update the path of the train and test datasets
* Visualising 9 different cancers 1 from each class
![skin cancer cnn github](https://user-images.githubusercontent.com/69252134/130804756-c6bc029a-3e90-44f3-8eb7-a30406397019.png)

## Model Building
---
I have started building sequential model with **Conv2D**, **MaxPool2D** only and trained on **20 epochs** just to visualize **overfitting** is present or not.
![overfitting cnn git hub](https://user-images.githubusercontent.com/69252134/130806093-172d6dfc-af74-483e-a4f5-80058f60e037.png)
There is clear sign of overfitting on the model to overcome this performed **data Augmentation** and added **Dropouts** and then trained on **20 epochs**
![reduced overfitting cnn git hub](https://user-images.githubusercontent.com/69252134/130806430-6fd82fae-5d8d-4281-9c65-2f888abf2afd.png)
here we can see the over fitting has reduced but not totally
Next tried to find any **class imbalance** present in the data by importing **glob**

## Model Performance
---
#### I have used
<br>**Optimizer:** **sgd**
<br>**Loss function: SparseCategoricalCrossentropy**
<br>Trained on **30 epochs**
#### Acheived
* **Training Accuracy: 0.94**
* **Validation Accuracy: 0.72**







