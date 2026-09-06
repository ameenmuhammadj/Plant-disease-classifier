#  Plant Disease Classifier

Identify plant disease from plant leaf images using a deep learning model of ResNet18 for multi-class plant disease classification.

The project is based on a custom CNN baseline and comparison with a pretrained ResNet18 model, with the addition of confidence-aware prediction and visual explainability based on Grad-CAM.

##  Live Demo

[Open Live Demo](https://plant-disease-classific.streamlit.app)

> Ideally, a single leaf of the desired plant showing as little background as possible should be uploaded.

---

##  Project Overview

Diseases of plants can have a major impact on the productivity and output of agriculture. Prompt detection of diseases will enable farmers and agricultural workers take the necessary actions.

The aim of this project is to create an image classification system that can be used to classify the images of plant leaves into 38 different plant disease categories and healthy category.

Two deep learning techniques have been applied:

1. The 1st model is a custom CNN and is used as a baseline model.
2. ResNet18 coupled with Transfer Learning — as the final model.

The ResNet18 model had an accuracy of 99.48% on the test set of PlantVillage, much higher than the custom CNN baseline.

---

##  Objectives

- Establish a classification system for plant diseases for multiple classes.
Train a custom CNN as a base line.
- Transfer learning with ResNet18.
- Discuss the performance of the two models.
- Assess the models based on accuracy, precision, recall and F1-score.
- Analyze model errors using a confusion matrix.
- Implement confidence-aware predictions.
- Apply Grad-CAM for explainability of model.
- Share the final model as a public web app with Streamlit.

---

##  Dataset

Project is utilizing the PlantVillage data set.

### Dataset Information

- Total images: 54,305
- Number of classes: 38
- Crop species: 14
- Training images: approximately 43,444
- Test images: 10,861

The data consists of images of normal and diseased plant leaves.

### Dataset Split

The original training portion was further divided into:

- Training: 85%
- Validation: 15%
- Test: Original PlantVillage test set

Reproducibility was achieved by setting a fixed seed, 42.

---

##  Data Preprocessing

All images were resized to:

224 × 224 pixels

Images were normalized, based on the statistics of ImageNet:

Mean:
[0.485, 0.456, 0.406]

Standard Deviation:
[0.229, 0.224, 0.225]

### Data Augmentation

The training data used:

- Random horizontal flipping
- The angle of rotation is random, up to 15°.
- Random brightness adjustment
- Random contrast adjustment

Augmentation was applied only to the training data.

In order to validate and test images, they were resized and normalized but not randomly augmented.

---

##  Model 1 — Custom CNN

The baseline model was a custom convolutional network.

### Architecture

Input Image
↓
Conv2D (3 → 32)
↓
ReLU + MaxPool
↓
Conv2D (32 → 64)
↓
ReLU + MaxPool
↓
Conv2D (64 → 128)
↓
ReLU + MaxPool
↓
Conv2D (128 → 256)
↓
ReLU + MaxPool
↓
Adaptive Average Pooling
↓
Fully Connected Layer
↓
Dropout
↓
38-Class Output

### Training Configuration

- Optimizer: Adam
- Learning rate: 0.001
The loss function is the Cross Entropy Loss.The Cross Entropy Loss is the loss function.
- Epochs: 10
- Batch size: 32

### Result

Test Accuracy: 93.86%

---

The second model is ResNet18 model using Transfer Learning.The second model is ResNet18 model with Transfer Learning.

A pretrained ResNet18 model was used as the final classification model.

A new fully connected layer for 38 output classes has been used to replace the original ResNet18 classification layer.

### Training Configuration

- Architecture: ResNet18
- Pretrained weights: ImageNet
- Optimizer: Adam
- Learning rate: 0.0001
- Cross Entropy Loss is also known as the loss function.
- Epochs: 8
- Batch size: 32

### Result

Test Accuracy: 99.48%

---

##  Model Comparison

|Worst      | Custom CNN | ResNet18 |
|-----------|-----------:|---------:|
| Accuracy  |   93.86%   |  99.48%  |
| Precision |   0.9400   |  0.9949  |
| Recall    |   0.9386   |  0.9948  |
| F1-Score  |   0.9377   |  0.9948  |

### Performance Improvement

The test accuracy increased by 1.5% with ResNet18:

+5.62 percentage points

This is a good example of how well a transfer learning approach can perform classification of plant diseases images, compared to training a relatively small CNN from scratch.

---

##  Error Analysis

The ResNet18 model had a total of 56 errors over the 10,861 test images.

The majority of errors were between disease categories that are visually similar.

The most common mistake made was:

True:
Corn — Cercospora Leaf Spot / Gray Leaf Spot

Predicted:
Corn — Northern Leaf Blight

Misclassified samples:
13

Similar categories of tomato and potato disease were also seen to have other errors.

This means that similar diseases are difficult to classify even with a good image classification model.

---

##  Confidence-Aware Prediction

The deployed app includes the predicted disease and the confidence score of the model.

A confidence level of:

0.95

The selected was validated with the help of validation data.

Predictions below this threshold will be shown as:

Low-confidence prediction — It might not be the predicted result.

It makes the application avoid showing the same prediction as all of them being equally certain.

But confidence doesn't equal correctness: sometimes neural networks can be confidently wrong.

---

##  Grad-CAM Explainability

The regions in the image that the model relied on for its prediction were visually interpreted using Grad-CAM.

The target layer for the final convolutional layer of ResNet18 was used.

Grad-CAM offers a visual interpretation of the model's prediction and makes the classification system more interpretable.

> While Grad-CAM identifies important parts of an image, it does not determine if the model is relying on the correct disease features that are biologically acceptable.

---

##  Deployment

The trained model was deployed as a Streamlit web app.

### Deployment Architecture

User
↓
Streamlit Web Application
↓
Image Preprocessing
↓
ResNet18 Model
↓
Prediction + Confidence
↓
Top-5 Predictions

More than the usual size limit of 10 megabytes of a web upload to GitHub, the model file has been uploaded separately to Hugging Face.

### Hugging Face Model Repository

mdameen/plant-disease-resnet18

The repository contains:

- plant_disease_resnet18.pth
- class_names.json
- config.json

---

##  Tech Stack

### Programming Language

- Python

### Deep Learning

- PyTorch
- Torchvision

### Machine Learning

- Scikit-learn

### Data Processing

- NumPy
- Pandas
- PIL

### Visualization

- Matplotlib

### Explainability

- Grad-CAM

### Deployment

- Streamlit
- Hugging Face Model Hub

### Development Environment

- Google Colab
- NVIDIA GPU

---

##  Project Structure

plant-disease-classifier/
│
├── app.py
├── requirements.txt
└── README.md

The trained model is available separately on Hugging Face.

---

##  Run Locally

Clone the repository:

```bash
git clone https://github.com/mdameen/plant-disease-classifier.git
cd plant-disease-classifier
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

##  Limitations

Although the model achieved **99.48% accuracy on the PlantVillage test set**, this result should not be interpreted as 99.48% accuracy on arbitrary real-world plant photographs.

The PlantVillage dataset mainly contains controlled-condition leaf images.

Real-world images may contain:

- Complex backgrounds
- Multiple leaves
- Fruits and stems
- Different lighting conditions
- Partial or occluded leaves
- Different camera qualities
- Different disease appearances

Testing with a real-world tomato leaf image showed that the model can produce an incorrect prediction when the image distribution differs substantially from the training data.

Therefore, the system should be considered an **image classification research/project prototype** rather than a definitive agricultural diagnostic system.

Another limitation is that the current train/validation split is image-level. Images originating from the same leaf can potentially be visually similar, so leaf-level grouping would provide a stricter evaluation.

---

##  Future Improvements

Possible improvements include:

- Training with more real-world field images.
- Using leaf-level grouped train/test splitting.
- Adding leaf segmentation to isolate the leaf from the background.
- Using stronger data augmentation.
- Fine-tuning larger architectures.
- Testing on external datasets.
- Adding automatic image quality checking.
- Improving uncertainty estimation.
- Adding additional explainability methods.
- Optimizing the model for faster inference.

---

##  Key Results

|     Model    | Test Accuracy |
|--------------|--------------:|
|  Custom CNN  |     93.86%    |
| **ResNet18** |  **99.48%**   |

**Improvement:** +5.62 percentage points

The results demonstrate that transfer learning with ResNet18 significantly improved classification performance over the custom CNN baseline on the PlantVillage test set.

---

##  Author

**Muhammad Ameen J**

Computer Science and Engineering — Artificial Intelligence & Machine Learning

---

##  Disclaimer

This project is developed for educational and research purposes.

Predictions should not be considered a substitute for professional agricultural diagnosis or expert assessment.
