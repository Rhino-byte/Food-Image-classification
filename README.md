# Enhancing Culinary Accuracy: Deep Learning Solutions for Food Image Classification

![Team Building Cooking Classes near Springfield MA](https://github.com/user-attachments/assets/60fd1028-06a5-40c3-b7af-ea6e4694bb34)


## Overview *
This project is a food image classification model built using TensorFlow and trained on the Food101 dataset. The goal is to develop a deep learning model capable of accurately classifying images into one of 101 different food categories. This project is particularly useful for applications in restaurant automation, diet tracking apps, and food delivery services that require image-based food recognition.

## Problem statement
Accurate food classification remains a challenge in the food industry, affecting restaurants, delivery platforms, and nutrition-tracking apps. Manual identification leads to errors in  menu categorization and automated checkouts.

- `An AI-powered food classification model can automate this process with high accuracy, reducing errors, improving user experience, and enhancing operational efficiency.`

## Project Goals
### Optimization Strategies
Improving accuracy beyond 78% is crucial for reliable food  Image identification. Advanced deep learning techniques, data augmentation, and model optimization can enhance classification performance, reducing errors in real-world applications.

### Impact on Automation and Service Efficiency
AI-powered food classification can streamline restaurant operations through automated checkouts, minimize human error, and improve service efficiency. In food delivery, it optimizes workflow, enhances order accuracy, and reduces delays.


### Scalability and Real-World Application
For widespread adoption, the model must handle diverse food categories efficiently. Ensuring accuracy in automated checkouts, menu categorization, and large-scale applications is key to real-world feasibility.



## Stakeholders
1. **Restaurants and Food Service Providers**: They will benefit from improved operational efficiency, reduced errors in order processing, and enhanced customer satisfaction.
Food Delivery Platforms

  > These platforms can leverage accurate food classification to streamline their logistics, improve menu categorization, automate checkouts, and enhance user experience.

2. **Consumers**: End-users will gain from more accurate  personalized dietary recommendations and a more seamless food ordering experience.

3. **Health and Nutrition Apps**:These applications can utilize the classification model to provide users with better insights into their dietary habits and nutritional intake.

4. **Data Scientists and AI Developers**:Professionals in this field will be engaged in developing and refining the classification model, contributing to advancements in AI technology.

## Beneficiaries
- **Consumers**
> They will experience improved accuracy in food selection

-  **Restaurants**
> Enhanced operational efficiency and reduced errors will lead to cost savings and improved customer loyalty.

- **Food Delivery Services**
> Improved accuracy in food classification will streamline operations, reduce delivery times, and enhance customer satisfaction.

- **Health Professionals**
> They can utilize accurate food classification data to provide better dietary advice and support to their clients.

- **Technology Providers**
> Companies developing AI solutions will benefit from the demand for advanced food classification technologies, leading to potential partnerships and revenue growth.

By addressing these business questions and engaging the identified stakeholders and beneficiaries, the project can create a significant impact on the food industry, enhancing both operational efficiency and consumer experience.

## Project Workflow

This project follows a structured approach to developing a food image classification model:

### Understanding the Problem

Before starting, we conducted research on why food classification is important. Applications include restaurant automation, food tracking apps, and dietary analysis.

### Loading the Dataset

Using TensorFlow Datasets (TFDS), we loaded the Food101 dataset and inspected its structure:
```
import tensorflow as tf
import tensorflow_datasets as tfds

(train_data, test_data), ds_info = tfds.load(
    name='food101',
    split=['train', 'validation'],
    shuffle_files=True,
    as_supervised=True, 
    with_info=True
)
```
### Exploring the Data

We visualized some sample images along with their class labels to understand the dataset better. We also checked the class distribution to ensure a balanced dataset.

```
import matplotlib.pyplot as plt
class_names = ds_info.features['label'].names

for image, label in train_data.take(5):
    plt.imshow(image)
    plt.title(class_names[label.numpy()])
    plt.show()
```

### Preprocessing the Data

To prepare the data for training, we normalized pixel values and batched the dataset:
```
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0  # Resize and normalize
    return image, label

train_data = train_data.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
test_data = test_data.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
```

5. Building the Model

We used a Convolutional Neural Network (CNN) for image classification. We started with a simple model and later fine-tuned it.
```
import tensorflow as tf
from  tensorflow.Keras import layers

# Finding our best base model to proceed with fine-tuning

input_shape = (224,224,3)
base_model = tf.keras.applications.EfficientNetB0(include_top = False)
base_model.trainable = False

# Creating a Functional API model
inputs = layers.Input(shape=input_shape,name = "input_layer")
# Since the Efficient models have rescaling built-in we will not include a layer for that
# x = preprocessing.Rescaling(1/255.)(x)

x = base_model(inputs,training=False) # Just to enforce no updating the model weights
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(len(class_names))(x)

# To make sure the output tensors are in float 32 for numerical stability
outputs = layers.Activation('softmax',dtype=tf.float32,name='Softmax_layer')(x)
model = tf.keras.Model(inputs,outputs)

# Compile model
model.compile(loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'],
              optimizer= tf.keras.optimizers.Adam(learning_rate= 0.001))

```

### Data Augmentation

Since real-world images may have different orientations, lighting, or occlusions, we applied data augmentation techniques.

```
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    zoom_range=0.4,
    horizontal_flip=True,
    shear_range=0.3
)
```
We visualized how data augmentation modifies images to improve model generalization.
```
for image, label in train_data.take(1):
    image = tf.expand_dims(image[0], 0)
    aug_iter = datagen.flow(image, batch_size=1)
    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(next(aug_iter)[0])
        plt.axis("off")
    plt.show()

```
### Training the Model

We trained the model on the dataset, tracking loss and accuracy for improvements.

`history = model.fit(train_data, validation_data=test_data, epochs=10)`

### Evaluating the Model

We trained the model using the preprocessed dataset and monitored the accuracy and loss over multiple epochs.
We evaluated the model on test data and checked the accuracy.

```
test_loss, test_acc = model.evaluate(test_data)
print("Test Accuracy:", test_acc)
```
### Making Predictions

After training, we tested the model with new food images to see how well it could classify them.

Transfer learning with EfficientNetB0 significantly improved accuracy compared to a basic CNN.

### Feature Enhancements

Transfer Learning: Use pre-trained models such as EfficientNet to improve classification accuracy.

Data Augmentation: Apply techniques like flipping, rotation, and brightness adjustments to enhance generalization.

### Results and Observations

The model achieved an accuracy of around 76% on the test dataset.

Some misclassifications occurred between visually similar food items.

Training with a pre-trained model (e.g., MobileNetV2) could improve accuracy.


