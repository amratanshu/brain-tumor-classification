# Brain Tumor Classifier
## Project Overview
This project involves developing a brain tumor classifier using deep learning techniques. The primary goal is to classify images of brain tumors into different categories based on the type of tumor. The dataset is annotated with bounding box coordinates, depicting where the tumor is.

## Dataset

The current dataset contains high-quality MRI images of brain tumors with detailed annotations. It includes a total of 5,249 MRI images divided into training and validation sets. Each image is annotated with bounding boxes in YOLO format, and labels corresponding to one of the four classes of brain tumors.

Classes:
- Class 0: Glioma
- Class 1: Meningioma
- Class 2: No Tumor
- Class 3: Pituitary

```
- Base Directory: `./dataset`
  - Training Directory: `./dataset/Train`
  - Testing Directory: `./dataset/Val`
```

[Link to Kaggle dataset](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)

![image](https://github.com/amratanshu/brain-tumor-classification/blob/main/readme-resources/sample-training-images-1.png)

## What We Are Doing
1. **Data Loading and Augmentation:**
   - The dataset is loaded from specified directories, and images are augmented using transformations such as resizing, random horizontal flipping, and random rotation.
   - Both the training and validation datasets are split, ensuring a proper distribution for model training and validation.

2. **Custom Dataset Classes:**
   - `BrainTumorDataset`: Handles the training and validation datasets, including image loading, transformations, and handling bounding boxes.
   - `BrainTumorTestDataset`: Manages the test dataset with image loading and transformations.

3. **Data Preparation:**
   - Images and corresponding labels are loaded and preprocessed.
   - Bounding boxes are drawn on the images for visualization using the label txt files which contain bounding box coodinates and x, y widths.

4. **Model Definition:**
   - A custom model, `BrainTumorClassifier`, based on the ResNet-50 architecture, is defined and modified to classify the images into the specified number of classes.
   - The model is trained using the CrossEntropyLoss function and the Adam optimizer.
   - A learning rate scheduler is used to adjust the learning rate based on the validation loss.

5. **Training and Validation:**
   - The model is trained on a train and validation split on images in the `./dataset/Train` folder
   - Training is done for a specified number of epochs = 100, and performance metrics such as loss and accuracy are calculated for both the training and validation datasets.
   - Also, an `Early Stopper` is also implemented with `patience = 10`, `min_delta = 0.001` - will stop if 10 epochs give out a loss within this delta range.
   - The scheduler adjusts the learning rate based on the validation loss to optimize the training process.

6. **Testing:**
   - The model is evaluated on the images in the `./dataset/Val` folder, and performance metrics such as test loss and accuracy are calculated.
   - Sample images from the test set are displayed with bounding boxes for visual verification.
  
![image](https://github.com/amratanshu/brain-tumor-classification/blob/main/readme-resources/sample-test-images.png)

7. **Saving the Model:**
   - The trained model is saved to a file for future use and deployment.

## Model Details
The `BrainTumorClassifier` model is based on the `ResNet-50` architecture, which consists of multiple layers and blocks designed to extract high-level features from images. The key components of the model include:

- **ResNet-50 Backbone:**
  - Pre-trained ResNet-50 model is used as the backbone.
  - The model consists of convolutional layers, batch normalization, ReLU activations, and identity connections (residual blocks) that enable deep learning without vanishing gradients.

- **Fully Connected Layer:**
  - The final fully connected layer is replaced to match the number of classes in the dataset.
  - The output layer provides the class probabilities for the input images.

The model is trained end-to-end, leveraging the pre-trained weights of ResNet-50 for feature extraction and fine-tuning the final layers for classification specific to brain tumor images.

The training process involves optimizing the model weights to minimize the cross-entropy loss, and the performance is evaluated using accuracy metrics. The learning rate scheduler helps in fine-tuning the learning process to achieve optimal results.

This comprehensive approach ensures that the classifier can accurately identify and classify brain tumors from medical images, contributing to improved diagnostic capabilities.

## Accuracy achieved
We achieved an accuracy of ... on the test dataset

![image](https://github.com/amratanshu/brain-tumor-classification/blob/main/readme-resources/loss-accuracy-epoch%20graph.png)
