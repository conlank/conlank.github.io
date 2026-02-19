---
layout: post
title: Machine Learning for Pneumonia Detection
subtitle: There's lots to learn!
# gh-repo: daattali/beautiful-jekyll
# gh-badge: [star, fork, /follow]
tags: [data-science]
comments: true
mathjax: true
---

I recently finished up my DataCamp Data Scientist certification and have started to do some small projects to keep myself fresh and show what I learned. I started looking more seriously for work and felt like data science was something that leveraged my biochemistry degree, in the way that a lot of the research I did involved designing experiments and assessing the meanings of their outcomes. I also did a decent amount of coding during my masters degree, both for automating image analysis and crunching bioinformatics data.

Hopefully those experiences will transfer over well to a job in data science, as I know my writing and presentation skills will be useful in communicating the results of future projects. Being able to talk about my research on Krabbe disease at a range of levels from discussing with other scientists to those with almost no science background was a super useful skill to learn and be confident with.

So without further adieu I would like to present a small project I did with machine learning.

# Machine Learning for Pneumonia Detection

## Intro

Every year approximately 4 million people die of pneumonia, and about 450 million people are affected by this disease. Pneumonia is the leading cause of death in developing countries and the very old, the very young and the chronically ill. Pneumonia is an inflammatory condition that impacts the alveoli of the lungs, the small structures where blood is oxygenated. This causes the symptoms of coughing, chest pain, and, difficulty breathing.

Pneumonia is often caused by infection by either viruses or bacteria, but determining the exact pathogen responsible is difficult.  Diagnosis is usually based on symptoms, with chest x-rays being used to confirm the diagnosis. Pneumonia is treated with antivirals or antibiotics depending on the cause, with supplemental oxygen therapy used if blood oxygen levels drop.

In this project, we will investigate the problem of confirming a pneumonia diagnosis using a chest x-ray and machine learning. I will use a public dataset of chest x-rays from Zhang et al, 2018. I will use two methods of training, in one method I will create a new classifier to label the images, and in another method I will use transfer learning from the publicly available image classifier VGG19 and the image net weights.

## Import Data and Libraries

First we will download the dataset, extract the images and create the testing and training sets.


```python
import os
import zipfile
import glob
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
%pprint ON
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
```

    Pretty printing has been turned OFF



```python
# Check for the zipped data, download it if it doesn't exist
if not os.path.isfile('chest-xray-pneumonia.zip'):
    print("Data has not been downloaded, acquiring the data for this notebook...")
    os.system("""
    curl -L -o chest-xray-pneumonia.zip https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia
    """)
    print("Data downloaded")

# Unzip the data
if not os.path.exists('chest_xray'):
    with zipfile.ZipFile('chest-xray-pneumonia.zip', 'r') as zip_ref:
        print("Extracting ...")
        zip_ref.extractall()
        print('Done')

data_dir = Path('chest_xray/chest_xray')
train_dir = data_dir/'train'
val_dir = data_dir/'val'
test_dir = data_dir/'test'
```


```python
# Create dataframes of training and test data for ease of data analysis and visualization

def df_from_path(foldr):
    filepath = []
    label = []

    folders = os.listdir(foldr)

    for folder in folders:
        f_path = os.path.join(foldr, folder)
        imgs = os.listdir(f_path)

        for img in imgs:
            img_path = os.path.join(f_path, img)
            filepath.append(img_path)
            label.append(folder)

    file_path_series = pd.Series(filepath, name = 'filepath')
    label_path_series = pd.Series(label, name = 'label')
    df_out = pd.concat([file_path_series, label_path_series], axis = 1)
    return df_out

df_train = df_from_path(train_dir)
df_test = df_from_path(test_dir)

print(f"The shape of the training set data is: {df_train.shape}")
print(f"The shape of the test set data is:     {df_test.shape}")
```

    The shape of the training set data is: (5216, 2)
    The shape of the test set data is:     (624, 2)


### Tools and Libraries

I will use pandas, seaborn, keras, and sci-kit learns to create the classification pipeline. Pandas is used to create dataframes of the training, validation, and testing images and labels. Seaborn is used for creating graphs and previewing the images. Keras is the main library used for the machine learning pipeline for model creation and evaluation. Sci-kit learns is used for assessing model metrics.


```python
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
# from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
```

### Data Set Qualities
This dataset is 5863 jpeg image files split between the pneumonia and normal categories. The images are 224x224 pixels, in greyscale. They were obtained from a retrospective cohorts of pediatric patients from Guangzhou Women and Children's Medical Center. We will use 4695 images for training, 521 images for training validation, and 624 images for testing our models. Some issues we need to be aware of with this dataset are the greyscale values, and uneven distribution of categories. To work around these issues we will perform image normalization to convert the greyscale values from 0-255 to 0.0-1.0. We will also perform data augmentation by random zoom, random rotation, horizontal flips, and random x and y translation. This will allow us to provide more images for training, which should increase the accuracy of the trained model.


```python
labels = ['NORMAL', 'PNEUMONIA']
image_size = (224, 224)
batch_size = 32

# Load datasets using Keras builtin module
print('Training Images:')
train = image_dataset_from_directory(
    directory = train_dir,
    labels = 'inferred',
    validation_split = 0.1,
    subset = 'training',
    # label_mode = 'binary',
    class_names = labels,
    image_size = image_size,
    batch_size = batch_size,
    color_mode = 'grayscale',
    seed = 356,
)
print(' ')
print('Validation Images:')
val = image_dataset_from_directory(
    directory = train_dir,
    labels = 'inferred',
    validation_split = 0.1,
    subset = 'validation',
    # label_mode = 'binary',
    class_names = labels,
    image_size = image_size,
    batch_size = batch_size,
    color_mode = 'grayscale',
    seed = 356,
)
print(' ')
print('Testing Images:')
test = image_dataset_from_directory(
    directory = test_dir,
    labels = 'inferred',
    # label_mode='binary',
    class_names = labels,
    image_size = image_size,
    batch_size = batch_size,
    color_mode = 'grayscale',
    seed = 356,
)
```

    Training Images:
    Found 5216 files belonging to 2 classes.
    Using 4695 files for training.


    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1771081785.809542 1938631 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 6017 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 3070, pci bus id: 0000:01:00.0, compute capability: 8.6



    Validation Images:
    Found 5216 files belonging to 2 classes.
    Using 521 files for validation.

    Testing Images:
    Found 624 files belonging to 2 classes.



```python
# Extract and encode class labels
train_labels = train.class_names
test_labels = test.class_names
val_labels = val.class_names

label_encoder = LabelEncoder()
label_encoder.fit(labels)
train_labels_encoded = label_encoder.transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)
val_labels_encoded = label_encoder.transform(val_labels)
```


```python
for img_batch, labels_batch in train:
    print(f"Shape of X_train : {img_batch.shape}")
    print(f"Shape of Y_train : {labels_batch.shape}")
    break
```

    Shape of X_train : (32, 224, 224, 1)
    Shape of Y_train : (32,)



```python
# Normalize pixel values of the data

train = train.map(lambda x, y: (x / 255.0, y))
test = test.map(lambda x, y: (x / 255.0, y))
val = val.map(lambda x, y: (x / 255.0, y))
```


```python
# Graph the data to see if there are any issues with class sizes
sns.set_style('dark')
sns.set_context('paper')
palette = sns.color_palette('viridis')
sns.set_palette(palette)

def plot_dist_count(df_in, dataset_name):
    count = df_in['label'].value_counts()

    fig, axs = plt.subplots(1, 2, figsize = (12, 6), facecolor = 'white')

    fig.suptitle(f'{dataset_name} Data')

    axs[0].pie(count, labels = count.index, autopct = '%1.1f%%', startangle = 120, palette = palette)
    axs[0].set_title('Distribution of Categories')

    sns.barplot(x = count.index, y = count.values, ax = axs[1], palette = palette)
    axs[1].set_title('Count of Categories')

    plt.tight_layout()
    plt.show()

plot_dist_count(df_train, 'Training')
plot_dist_count(df_test, 'Testing')
```



![png](/assets/img/2026-02-19-doing-ml-classification/output_13_0.png)





![png](/assets/img/2026-02-19-doing-ml-classification/output_13_1.png)




```python
# Show some example images for each class in the training and testing datasets

def show_images(path, labels, num_images = 5):
    data_set_name = os.path.split(path)[1]

    fig, axs = plt.subplots(2, num_images, figsize = (15, 6), facecolor = 'white')
    fig.suptitle(f'{data_set_name.title()}ing Images')

    for j, label in enumerate(labels):
        new_path = path/label
        img_filenames = os.listdir(new_path)
        num_images = min(num_images, len(img_filenames))

        for i, img_filename in enumerate(img_filenames[:num_images]):
            img_path = os.path.join(new_path, img_filename)
            # print(img_path)
            img = mpimg.imread(img_path)
            # print(img)

            axs[j][i].imshow(img, cmap='gray')
            axs[j][i].axis('image')
            axs[j][i].set_xticks([])
            axs[j][i].set_yticks([])
            axs[j][i].set_title(img_filename)
            # axs[j][i].set_ylabel(label)

        axs[j][0].set_ylabel(label)
    fig.align_labels()
    plt.tight_layout()
    plt.show()

show_images(train_dir, labels)
show_images(test_dir, labels)
```



![png](/assets/img/2026-02-19-doing-ml-classification/output_14_0.png)





![png](/assets/img/2026-02-19-doing-ml-classification/output_14_1.png)




```python
# Define the data augmentation that will be used to make the dataset larger

data_augmentation_layers = [
    RandomFlip('horizontal'),
    RandomRotation(0.3),
    RandomZoom(
        height_factor = 0.2,
        width_factor = 0.2
    ),
    RandomTranslation(
        height_factor = 0.1,
        width_factor = 0.1,
    )
]


def data_augmentation(m):
    for layer in data_augmentation_layers:
        m.add(layer)
```



### Model Creation


```python
input_shape = (224, 224, 1)

model = Sequential()
model.add(keras.Input(shape = input_shape))
data_augmentation(model)
model.add(Conv2D(32, (3, 3), strides = 1, padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides = 2, padding = 'same'))
model.add(Conv2D(64, (3, 3), strides = 1, padding = 'same', activation = 'relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides = 2, padding = 'same'))
model.add(Conv2D(64, (3, 3), strides = 1, padding = 'same', activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides = 2, padding = 'same'))
model.add(Conv2D(128, (3, 3), strides = 1, padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2, 2), strides = 2, padding = 'same'))
model.add(Conv2D(256, (3, 3), strides = 1, padding = 'same', activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(units = 128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy', 'recall', 'precision'])
model.summary(show_trainable = True)

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_2"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                </span>┃<span style="font-weight: bold"> Output Shape          </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Trai… </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│ random_flip (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomFlip</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ random_rotation             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomRotation</span>)            │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ random_zoom (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomZoom</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ random_translation          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomTranslation</span>)         │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ conv2d_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">320</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ batch_normalization_6       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">128</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)        │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ max_pooling2d_4             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)              │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ conv2d_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)  │     <span style="color: #00af00; text-decoration-color: #00af00">18,496</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dropout_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)  │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ batch_normalization_7       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">112</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)  │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)        │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ max_pooling2d_5             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)              │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ conv2d_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)    │     <span style="color: #00af00; text-decoration-color: #00af00">36,928</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ batch_normalization_8       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">56</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)    │        <span style="color: #00af00; text-decoration-color: #00af00">256</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)        │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ max_pooling2d_6             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)    │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)              │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ conv2d_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)           │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │     <span style="color: #00af00; text-decoration-color: #00af00">73,856</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dropout_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ batch_normalization_9       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">28</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │        <span style="color: #00af00; text-decoration-color: #00af00">512</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)        │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ max_pooling2d_7             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling2D</span>)              │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ conv2d_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)   │    <span style="color: #00af00; text-decoration-color: #00af00">295,168</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dropout_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ batch_normalization_10      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">14</span>, <span style="color: #00af00; text-decoration-color: #00af00">256</span>)   │      <span style="color: #00af00; text-decoration-color: #00af00">1,024</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)        │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ flatten_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">50176</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dense_5 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │  <span style="color: #00af00; text-decoration-color: #00af00">6,422,656</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dropout_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)           │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dense_6 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">129</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
└─────────────────────────────┴───────────────────────┴────────────┴───────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,849,729</span> (26.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">6,848,641</span> (26.13 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">1,088</span> (4.25 KB)
</pre>



### Train Model

We use learning rate reduction to prevent over fitting the data set, we also set up also set up early stopping to prevent over fitting. Reducing the learning rate when the validation accuracy platues helps prevent overfitting by reducing the change in the model weights when the validation accuracy is stagnant. Early stopping prevents overfitting by minimizing validation loss, which is an indicator how well the model correctly predicted the validation data.


```python
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy', patience = 4, verbose = 1, factor = 0.3, min_lr = 0.0001)

early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 9, verbose = 1, restore_best_weights = True, start_from_epoch = 5)
```


```python
history = model.fit(x = train, epochs = 50, validation_data = val, callbacks =  [learning_rate_reduction, early_stopping])
# history = model.fit(x = train, epochs = 50, validation_data = val, callbacks =  [learning_rate_reduction])
```

    Epoch 1/50


    E0000 00:00:1771088664.719300 1938631 meta_optimizer.cc:967] layout failed: INVALID_ARGUMENT: Size of values 0 does not match size of permutation 4 @ fanin shape inStatefulPartitionedCall/sequential_2_1/dropout_6_1/stateless_dropout/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer


    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m14s[0m 75ms/step - accuracy: 0.7966 - loss: 0.8183 - precision: 0.8786 - recall: 0.8426 - val_accuracy: 0.7447 - val_loss: 11.9466 - val_precision: 0.7447 - val_recall: 1.0000 - learning_rate: 0.0010
    Epoch 2/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.8445 - loss: 0.3499 - precision: 0.9128 - recall: 0.8741 - val_accuracy: 0.7447 - val_loss: 26.5729 - val_precision: 0.7447 - val_recall: 1.0000 - learning_rate: 0.0010
    Epoch 3/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 73ms/step - accuracy: 0.8705 - loss: 0.3284 - precision: 0.9268 - recall: 0.8965 - val_accuracy: 0.7447 - val_loss: 52.4052 - val_precision: 0.7447 - val_recall: 1.0000 - learning_rate: 0.0010
    Epoch 4/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.8922 - loss: 0.2681 - precision: 0.9388 - recall: 0.9145 - val_accuracy: 0.7889 - val_loss: 1.0963 - val_precision: 0.7825 - val_recall: 0.9923 - learning_rate: 0.0010
    Epoch 5/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 72ms/step - accuracy: 0.8948 - loss: 0.2960 - precision: 0.9450 - recall: 0.9114 - val_accuracy: 0.7466 - val_loss: 2.0088 - val_precision: 0.7462 - val_recall: 1.0000 - learning_rate: 0.0010
    Epoch 6/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9112 - loss: 0.2389 - precision: 0.9507 - recall: 0.9286 - val_accuracy: 0.9443 - val_loss: 0.1483 - val_precision: 0.9761 - val_recall: 0.9485 - learning_rate: 0.0010
    Epoch 7/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 71ms/step - accuracy: 0.9165 - loss: 0.2358 - precision: 0.9469 - recall: 0.9403 - val_accuracy: 0.2802 - val_loss: 10.9650 - val_precision: 0.5251 - val_recall: 0.3505 - learning_rate: 0.0010
    Epoch 8/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9161 - loss: 0.2228 - precision: 0.9494 - recall: 0.9369 - val_accuracy: 0.3282 - val_loss: 3.2287 - val_precision: 1.0000 - val_recall: 0.0979 - learning_rate: 0.0010
    Epoch 9/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9171 - loss: 0.2218 - precision: 0.9556 - recall: 0.9317 - val_accuracy: 0.7639 - val_loss: 1.1281 - val_precision: 0.7593 - val_recall: 1.0000 - learning_rate: 0.0010
    Epoch 10/50
    [1m146/147[0m [32m━━━━━━━━━━━━━━━━━━━[0m[37m━[0m [1m0s[0m 68ms/step - accuracy: 0.9250 - loss: 0.2147 - precision: 0.9512 - recall: 0.9471
    Epoch 10: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9255 - loss: 0.2092 - precision: 0.9545 - recall: 0.9447 - val_accuracy: 0.9194 - val_loss: 0.2454 - val_precision: 0.9701 - val_recall: 0.9201 - learning_rate: 0.0010
    Epoch 11/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9408 - loss: 0.1670 - precision: 0.9647 - recall: 0.9553 - val_accuracy: 0.8944 - val_loss: 0.2927 - val_precision: 0.8776 - val_recall: 0.9974 - learning_rate: 3.0000e-04
    Epoch 12/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m20s[0m 72ms/step - accuracy: 0.9444 - loss: 0.1577 - precision: 0.9643 - recall: 0.9607 - val_accuracy: 0.9386 - val_loss: 0.1925 - val_precision: 0.9495 - val_recall: 0.9691 - learning_rate: 3.0000e-04
    Epoch 13/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9468 - loss: 0.1518 - precision: 0.9660 - recall: 0.9621 - val_accuracy: 0.9136 - val_loss: 0.1723 - val_precision: 0.9914 - val_recall: 0.8918 - learning_rate: 3.0000e-04
    Epoch 14/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9414 - loss: 0.1496 - precision: 0.9628 - recall: 0.9581 - val_accuracy: 0.9463 - val_loss: 0.1504 - val_precision: 0.9891 - val_recall: 0.9381 - learning_rate: 3.0000e-04
    Epoch 15/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m11s[0m 72ms/step - accuracy: 0.9408 - loss: 0.1554 - precision: 0.9641 - recall: 0.9558 - val_accuracy: 0.9347 - val_loss: 0.1698 - val_precision: 0.9836 - val_recall: 0.9278 - learning_rate: 3.0000e-04
    Epoch 15: early stopping
    Restoring model weights from the end of the best epoch: 6.


### Transfer Learning

In transfer learning a previously trained model is applied to a different but related task. This allows one to use a pre-trained model as a starting point and allows features learned during training on one task to be applied to a new task. This allows for better performance with less data and training, as your model is not starting from zero. Transfer learning involves fine-tuning the pre-trained model by changing the final layers for the new task, allowing the base layers to be reused. This allows knowledge gained during previous, such as image recognition, to quickly and efficiently be generalized to a new task.


```python
base_model = keras.applications.VGG19(weights = 'imagenet', include_top = False)
base_model.trainable = False

t_model = Sequential()
t_model.add(keras.Input(shape = (224, 224, 1)))
data_augmentation(t_model)
t_model.add(Conv2D(3, (3, 3), strides = 1, padding = 'same', activation = 'relu'))
# t_model.add(keras.layers.Reshape((-1, 3)))
t_model.add(base_model)
t_model.add(BatchNormalization())
t_model.add(Flatten())
t_model.add(Dropout(0.45))
t_model.add(Dense(800, activation = 'relu'))
t_model.add(Dropout(0.25))
t_model.add(Dense(130, activation = 'relu'))
t_model.add(Dense(1, activation = 'sigmoid'))
t_model.compile(optimizer=keras.optimizers.Adamax(learning_rate=0.001), loss = 'binary_crossentropy', metrics = ['accuracy', 'recall', 'precision'])
t_model.summary(show_trainable = True)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential_3"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                </span>┃<span style="font-weight: bold"> Output Shape          </span>┃<span style="font-weight: bold">    Param # </span>┃<span style="font-weight: bold"> Trai… </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━┩
│ random_flip (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomFlip</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ random_rotation             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomRotation</span>)            │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ random_zoom (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomZoom</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ random_translation          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)   │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">RandomTranslation</span>)         │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ conv2d_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv2D</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">224</span>, <span style="color: #00af00; text-decoration-color: #00af00">3</span>)   │         <span style="color: #00af00; text-decoration-color: #00af00">30</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ vgg19 (<span style="color: #0087ff; text-decoration-color: #0087ff">Functional</span>)          │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)     │ <span style="color: #00af00; text-decoration-color: #00af00">20,024,384</span> │   <span style="color: #ff0000; text-decoration-color: #ff0000; font-weight: bold">N</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ batch_normalization_11      │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">7</span>, <span style="color: #00af00; text-decoration-color: #00af00">512</span>)     │      <span style="color: #00af00; text-decoration-color: #00af00">2,048</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)        │                       │            │       │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ flatten_3 (<span style="color: #0087ff; text-decoration-color: #0087ff">Flatten</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">25088</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dropout_10 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">25088</span>)         │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dense_7 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">800</span>)           │ <span style="color: #00af00; text-decoration-color: #00af00">20,071,200</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dropout_11 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">800</span>)           │          <span style="color: #00af00; text-decoration-color: #00af00">0</span> │   <span style="font-weight: bold">-</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dense_8 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">130</span>)           │    <span style="color: #00af00; text-decoration-color: #00af00">104,130</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
├─────────────────────────────┼───────────────────────┼────────────┼───────┤
│ dense_9 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">1</span>)             │        <span style="color: #00af00; text-decoration-color: #00af00">131</span> │   <span style="color: #00af00; text-decoration-color: #00af00; font-weight: bold">Y</span>   │
└─────────────────────────────┴───────────────────────┴────────────┴───────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">40,201,923</span> (153.36 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">20,176,515</span> (76.97 MB)
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">20,025,408</span> (76.39 MB)
</pre>




```python
t_history = t_model.fit(x = train, epochs = 50, validation_data = val, callbacks =  [learning_rate_reduction, early_stopping])
# t_history = t_model.fit(x = train, epochs = 50, validation_data = val, callbacks =  [learning_rate_reduction])
```

    Epoch 1/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m29s[0m 183ms/step - accuracy: 0.8106 - loss: 0.5273 - precision: 0.8659 - recall: 0.8816 - val_accuracy: 0.6583 - val_loss: 0.6006 - val_precision: 1.0000 - val_recall: 0.5412 - learning_rate: 0.0010
    Epoch 2/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.8543 - loss: 0.3262 - precision: 0.8938 - recall: 0.9122 - val_accuracy: 0.7370 - val_loss: 0.5200 - val_precision: 0.9883 - val_recall: 0.6546 - learning_rate: 0.0010
    Epoch 3/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.8662 - loss: 0.3024 - precision: 0.9053 - recall: 0.9157 - val_accuracy: 0.7716 - val_loss: 0.4606 - val_precision: 0.9927 - val_recall: 0.6985 - learning_rate: 0.0010
    Epoch 4/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 170ms/step - accuracy: 0.8712 - loss: 0.2904 - precision: 0.9061 - recall: 0.9200
    Epoch 4: ReduceLROnPlateau reducing learning rate to 0.0003000000142492354.
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.8673 - loss: 0.2951 - precision: 0.9034 - recall: 0.9197 - val_accuracy: 0.8964 - val_loss: 0.2501 - val_precision: 0.9395 - val_recall: 0.9201 - learning_rate: 0.0010
    Epoch 5/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.8946 - loss: 0.2527 - precision: 0.9269 - recall: 0.9315 - val_accuracy: 0.8964 - val_loss: 0.2648 - val_precision: 0.9718 - val_recall: 0.8866 - learning_rate: 3.0000e-04
    Epoch 6/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.8901 - loss: 0.2533 - precision: 0.9233 - recall: 0.9292 - val_accuracy: 0.8618 - val_loss: 0.3035 - val_precision: 0.9907 - val_recall: 0.8222 - learning_rate: 3.0000e-04
    Epoch 7/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.8931 - loss: 0.2469 - precision: 0.9249 - recall: 0.9317 - val_accuracy: 0.8887 - val_loss: 0.2632 - val_precision: 0.9853 - val_recall: 0.8634 - learning_rate: 3.0000e-04
    Epoch 8/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 170ms/step - accuracy: 0.8982 - loss: 0.2331 - precision: 0.9311 - recall: 0.9318
    Epoch 8: ReduceLROnPlateau reducing learning rate to 0.0001.
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.9005 - loss: 0.2349 - precision: 0.9334 - recall: 0.9326 - val_accuracy: 0.9060 - val_loss: 0.2380 - val_precision: 0.9748 - val_recall: 0.8969 - learning_rate: 3.0000e-04
    Epoch 9/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.9050 - loss: 0.2256 - precision: 0.9328 - recall: 0.9398 - val_accuracy: 0.8618 - val_loss: 0.2984 - val_precision: 0.9847 - val_recall: 0.8273 - learning_rate: 1.0000e-04
    Epoch 10/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.9084 - loss: 0.2214 - precision: 0.9378 - recall: 0.9389 - val_accuracy: 0.8714 - val_loss: 0.2717 - val_precision: 0.9791 - val_recall: 0.8454 - learning_rate: 1.0000e-04
    Epoch 11/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.9067 - loss: 0.2198 - precision: 0.9387 - recall: 0.9355 - val_accuracy: 0.8887 - val_loss: 0.2510 - val_precision: 0.9797 - val_recall: 0.8686 - learning_rate: 1.0000e-04
    Epoch 12/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.9082 - loss: 0.2194 - precision: 0.9368 - recall: 0.9398 - val_accuracy: 0.8772 - val_loss: 0.2751 - val_precision: 0.9821 - val_recall: 0.8505 - learning_rate: 1.0000e-04
    Epoch 13/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.9067 - loss: 0.2203 - precision: 0.9379 - recall: 0.9363 - val_accuracy: 0.8848 - val_loss: 0.2492 - val_precision: 0.9795 - val_recall: 0.8634 - learning_rate: 1.0000e-04
    Epoch 14/50
    [1m147/147[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m27s[0m 181ms/step - accuracy: 0.9112 - loss: 0.2118 - precision: 0.9396 - recall: 0.9409 - val_accuracy: 0.8887 - val_loss: 0.2578 - val_precision: 0.9882 - val_recall: 0.8608 - learning_rate: 1.0000e-04
    Epoch 14: early stopping
    Restoring model weights from the end of the best epoch: 6.



```python
print("Small CNN results: ")
test_results = model.evaluate(test)
print(f"\nLoss of the model is      {test_results[0]:.4f}")
print(f"Accuracy of the model is  {test_results[1]*100:.2f}%")
print("\n\n\n")
print("Transfer Learning VGG19 model results: ")
t_test_results = t_model.evaluate(test)
print(f"\nLoss of the model is      {t_test_results[0]:.4f}")
print(f"Accuracy of the model is  {t_test_results[1]*100:.2f}%")
```

    Small CNN results:
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 22ms/step - accuracy: 0.8910 - loss: 0.3094 - precision: 0.8626 - recall: 0.9821

    Loss of the model is      0.3094
    Accuracy of the model is  89.10%




    Transfer Learning VGG19 model results:
    [1m20/20[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 91ms/step - accuracy: 0.8846 - loss: 0.2918 - precision: 0.9251 - recall: 0.8872

    Loss of the model is      0.2918
    Accuracy of the model is  88.46%

### Post Training Analysis

```python
def best_epoch(training_history):
    return training_history.history['val_accuracy'].index(max(training_history.history['val_accuracy'])) + 1

def plot_training_data(train_hist):

    # plt.style.use('seaborn-darkgrid')
    metrics = ['accuracy', 'loss', 'recall', 'precision']
    fig, axs = plt.subplots(1, len(metrics), figsize = (16, 6))
    best_e = best_epoch(train_hist)

    for i, metric in enumerate(metrics):
        axs[i].plot(train_hist.history[metric], label = f"Training {metric.title()}", color = 'blue')
        axs[i].plot(train_hist.history[f"val_{metric}"], label = f"Validation {metric.title()}", color = 'red')
        axs[i].scatter(best_e - 1, train_hist.history[f"val_{metric}"][best_e - 1], color = 'green', label = f"Best Epoch: {best_e}")
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(f"{metric.title()}")
        axs[i].set_title(f"Training and Validation {metric.title()}")
        axs[i].legend()

    plt.tight_layout()
    plt.show()

print("Small CNN")
plot_training_data(history)
print("Transfer Learning")
plot_training_data(t_history)
```

    Small CNN




![png](/assets/img/2026-02-19-doing-ml-classification/output_27_1.png)



    Transfer Learning




![png](/assets/img/2026-02-19-doing-ml-classification/output_27_3.png)



#### Confusion Matrix


```python
def conf_mat(model):
    y_true = []
    y_pred = []

    for x,y in test:
        y = tf.concat(y, axis = 1)
    # print(y.numpy())
        y_true.append(y)
        y_pred.append((model.predict(x) > 0.5).astype('int32'))

# print(y_true)
# print(y_pred)
    y_pred = tf.concat(y_pred, axis = 0)
    y_true = tf.concat(y_true, axis = 0)

    cm = confusion_matrix(y_true, y_pred, normalize = 'true')

    plt.figure(figsize=(6,6))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    sns.heatmap(cm, annot=True, cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.title(f"Confusion Matrix")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    print(classification_report(y_true, y_pred, target_names = ['Healthy', 'Pneumonia']))

print('Confusion Matrix for Small CNN')
conf_mat(model)
```

    Confusion Matrix for Small CNN
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 69ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 57ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 39ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 40ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 38ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 33ms/step




![png](/assets/img/2026-02-19-doing-ml-classification/output_29_1.png)



                  precision    recall  f1-score   support

         Healthy       0.96      0.74      0.84       234
       Pneumonia       0.86      0.98      0.92       390

        accuracy                           0.89       624
       macro avg       0.91      0.86      0.88       624
    weighted avg       0.90      0.89      0.89       624



From the confusion matrix for the small CNN, we can see the false positive and false negative rates. Our false positive situation, where a healthy individual's x-ray is predicted to have pneumonia occurs 26% of the time. Although the false positive rate seems high, the main focus of a diagnostic test is to minimize false negative results, as those are the cases where there is the most potential harm.


```python
print('Confusion Matrix for Transfer Learning CNN')
conf_mat(t_model)
```

    Confusion Matrix for Transfer Learning CNN
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 152ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 117ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 117ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 118ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 121ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 115ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 116ms/step
    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 75ms/step




![png](/assets/img/2026-02-19-doing-ml-classification/output_31_1.png)



                  precision    recall  f1-score   support

         Healthy       0.82      0.88      0.85       234
       Pneumonia       0.93      0.89      0.91       390

        accuracy                           0.88       624
       macro avg       0.87      0.88      0.88       624
    weighted avg       0.89      0.88      0.89       624



In the transfer learning CNN we achieved more balanced results between the false negative rate (11%) and the false positive rate (12%). Although these more balanced results show the models learning both classes equally, this is not ideal for the use case of a medical diagnostic tool. With a diagnostic test, there aim is to have as low of a false negative rate as possible, as false positives will almost certainly be caught during a confirmation test.
