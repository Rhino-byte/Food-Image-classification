# Enhancing Culinary Accuracy: Deep Learning Solutions for Food Image Classification

![Team Building Cooking Classes near Springfield MA](https://github.com/user-attachments/assets/60fd1028-06a5-40c3-b7af-ea6e4694bb34)

**Authors:**

[Noel Christopher](https://github.com/NOE0464) 

[Savins Nanyaemuny](https://github.com/Rhino-byte)

[Anthony Ekeno](https://github.com/sananthonio)

[Linet Lydia](https://github.com/LinetLydia)

[Imran Mahfoudh](https://github.com/malvadaox)


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

### Building the Model

We used a Convolutional Neural Network (CNN) architecture for image classification. We started with a simple model and later fine-tuned it.
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

Data augmentation (e.g., flipping, rotating, adjusting brightness) exposed the model to varied scenarios, improving its ability to recognize foods in real-world settings. 
Finally, prefetching was implemented to prepare the next batch of data while processing the current one, reducing delays and optimizing training speed.
Together, these steps created a robust and efficient training pipeline for high-accuracy food classification.

![Screenshot 2025-02-26 061801](https://github.com/user-attachments/assets/f8ad7280-8743-4f02-b22e-d8175741a77d)


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

**Comparing History Curves:** we tracked the loss and accuracy curves for training and testing datasets throughout training. By comparing these curves, we gained insights into:
- **Model Performance:** A steadily decreasing loss and increasing accuracy indicated effective learning.
- **Overfitting Signs:** If the training accuracy improved while test accuracy plateaued or declined, it signaled overfitting, helping us adjust the model accordingly.
- **Model Choice:** The model with the best balance between train and test performance—achieving high accuracy with low overfitting—was chosen as the best-performing model.

![Screenshot 2025-02-26 064543](https://github.com/user-attachments/assets/3248ac78-9916-49f7-a9db-93a48a4d4cf5)

#### Addressing Overfitting

To mitigate overfitting in our Food Vision 101 model, we implemented several strategies to enhance the model’s generalization capability, including:
1. `Data Augmentation:` We applied techniques like random rotations, flips, shifts, and zooms to artificially expand the training dataset, helping the model learn robust features rather than memorizing specific images.
2. `Early Stopping:` We monitored the validation loss and halted training when it stopped improving, preventing the model from over-training on the training data.
3. `Regularization Techniques:` We incorporated Dropout layers, which randomly deactivate neurons during training, reducing the risk of the model becoming too reliant on specific pathways.
4. `Transfer Learning with Fine-Tuning:` Initially, we used feature extraction to leverage pre-trained weights without unfreezing layers. During fine-tuning, only the top layers were updated, maintaining the general features learned from the base model and minimizing overfitting risks.
5. `Batch Normalization:` Implemented batch normalization layers to stabilize training, allowing the model to learn faster and generalize better.
6. `Increasing Training Data:` By performing different experiments while tracking how the model performance was affected by the data. [Weight & Biases](https://api.wandb.ai/links/savins-nanyaemuny-moringa-school/h0xdx4n8)






### Evaluating the Model

We trained the model using the preprocessed dataset and monitored the accuracy and loss over multiple epochs.
We evaluated the model on test data and checked the accuracy.
To evaluate our Food Vision 101 model's performance in real-world scenarios, we tested it on custom images outside of the training and validation datasets. This approach allowed us to observe how well the model generalized to new, unseen food images.

![Screenshot 2025-02-26 071409](https://github.com/user-attachments/assets/a936f6d0-bc58-4e5e-81ea-441919be8017)

The F1-score visualization for the 101 different food classes provides a detailed view of the model's performance across each category. The F1 score combines precision and recall, offering a balanced measure of model accuracy in classifying each food type.
From the chart, it is evident that certain classes, such as edamame, macarons, and oysters, achieved higher F1 scores, indicating the model's strong performance in accurately predicting these categories. This could be due to distinctive features in the images or a balanced representation of the training data.

![Screenshot 2025-02-26 065430](https://github.com/user-attachments/assets/2042c8d7-ee6c-4c1e-b172-54ad1288c3f5)


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

# Success Story
To evaluate our Food Vision 101 model's performance in real-world scenarios, we tested it on custom images outside of the training and validation datasets. This approach allowed us to observe how well the model generalized to new, unseen food images
Checkout the [Application](https://huggingface.co/spaces/bushman254/FoodImageClassifier)


# Recommendations
**Enhance Business Framing:** Clearly defining project goals and intended use cases ensures that the model aligns with business needs and user expectations. This clarity improves communication with stakeholders and increases project adoption.

**Improve Model Explainability:** Implementing SHAP/LIME, confusion matrices, and misclassification analysis helps in understanding how the model makes predictions. This transparency increases trust in the model, facilitates debugging, and allows stakeholders to make informed decisions based on the model’s outputs.

**Address Class Imbalances:** Using weighted loss functions or data augmentation mitigates the impact of underrepresented classes in the dataset. This step improves overall model performance, ensuring fair and accurate classifications across all food categories.

**Optimize for Deployment:** Including API integration or inference pipelines makes the model ready for real-world applications. This step ensures smooth integration into food-related platforms, improving usability and increasing the model’s impact.

# Conclusions
The project effectively demonstrates food classification using machine learning.

While technically strong, adding interpretability and business context will make it more impactful.

The next steps should focus on refining model deployment and ensuring usability in its intended application.nclusions


