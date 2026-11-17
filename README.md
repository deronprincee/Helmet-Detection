
# Helmet Compliance Classification Using Machine Learning

A machine learning project that classifies images into two categories: **helmet** and **no_helmet**, using image feature extraction and classical machine learning algorithms.

Developed as part of the **Applied Machine Learning** module at **Teesside University**, this project explores how computer vision techniques can support the identification of potential safety helmet violations in construction-related images.

The project was completed collaboratively, with a shared image preprocessing and feature extraction pipeline. My individual contribution focused on developing, tuning and evaluating a **Support Vector Machine (SVM)** classifier using R.

## Project Overview

Safety helmets are an important part of personal protective equipment (PPE) on construction sites. Identifying workers who are not wearing helmets can help improve safety monitoring.

The objective of this project was to investigate whether traditional machine learning techniques could distinguish between images containing helmets and those without helmets.

Rather than using a deep learning object detection system, we approached the problem as a **binary image classification task**, allowing multiple machine learning algorithms to be compared using the same dataset and extracted features.

### Key Features

- Binary image classification: `helmet` vs `no_helmet`
- Image preprocessing and standardisation
- Feature extraction using HOG, spatial uLBP and HSV
- Linear Support Vector Machine classification
- Hyperparameter tuning using 5-fold cross-validation
- Evaluation using accuracy, precision, recall, F1-score and confusion matrices
- Comparison against alternative machine learning models

## Technologies Used

| Technology | Purpose |
|---|---|
| R | Model development and evaluation |
| RStudio | Development environment |
| Linear SVM | Main classification algorithm |
| RBF SVM | Alternative SVM model comparison |
| HOG | Gradient and shape feature extraction |
| Spatial uLBP | Texture feature extraction |
| HSV | Colour feature extraction |
| CSV | Storage of extracted numerical features |
| Cross-validation | Model selection and hyperparameter tuning |

## Dataset

The dataset was assembled from three publicly available Kaggle sources:

1. [PPE Kit Detection (Construction Site Workers)](https://www.kaggle.com/datasets/ketakichalke/ppe-kit-detection-construction-site-workers/data)
2. [Human Images Dataset – Men and Women](https://www.kaggle.com/datasets/snmahsa/human-images-dataset-men-and-women)
3. [Safety Helmet Detection](https://www.kaggle.com/code/rockatoo/safety-helmet-detection/notebook)

The original combined dataset contained over 3,000 images. After dataset selection, cleaning and preparation, the final dataset contained **1,200 images**.

### Dataset Distribution

| Class | Images |
|---|---:|
| Helmet | 600 |
| No Helmet | 600 |
| **Total** | **1,200** |

The dataset was divided using a reproducible, stratified 70/30 train-test split.

| Split | Images |
|---|---:|
| Training | 840 |
| Testing | 360 |

The balanced dataset allowed the models to be evaluated without one class dominating the results.

## Methodology

### 1. Image Preprocessing

The shared preprocessing pipeline prepared images for machine learning.

Steps included:

- Selecting images belonging to the two target classes
- Checking image validity and duplicates
- Resizing images to **128 × 128 pixels**
- Creating a stratified training and testing split
- Extracting numerical features
- Exporting the training and testing features to CSV files

The resulting files were:

- `features_train.csv`
- `features_test.csv`

These datasets were shared across the different classifiers to allow a fair comparison.

### 2. Feature Extraction

Instead of training directly on raw image pixels, the project used handcrafted image features.

Each image was represented by a **396-dimensional feature vector**.

| Feature Type | Features | Description |
|---|---:|---|
| Spatial uniform Local Binary Patterns (uLBP) | 236 | Captures local texture patterns |
| Histogram of Oriented Gradients (HOG) | 144 | Captures edge directions and shape information |
| HSV Colour Features | 16 | Captures colour characteristics |
| **Total** | **396** | |

Combining these features provided information about image texture, shape and colour.

### 3. Data Preparation for Modelling

Before training the SVM classifier:

- Non-feature columns were removed from the model input.
- Feature values were converted to numeric format.
- Non-finite values were treated as missing.
- Missing values were replaced using median values calculated from the training data.
- Features were scaled using statistics from the training data.

Imputation and scaling were fitted using training data only to reduce the risk of **data leakage**.

### 4. Linear SVM Training

My individual model was a **Linear Support Vector Machine**.

The classifier was trained using the shared engineered features and evaluated using **5-fold cross-validation**.

The following cost values were tested:

`0.01, 0.1, 1, 5, 10`

The best-performing Linear SVM configuration was:

- **Kernel:** Linear
- **Cost (C):** 0.01
- **Mean cross-validation accuracy:** 83.69%

An RBF SVM was also tested to explore whether a nonlinear decision boundary would improve classification performance.

## Results

The final Linear SVM was evaluated on the held-out test dataset of 360 images.

### Linear SVM Performance

| Metric | Result |
|---|---:|
| Accuracy | **85.3%** |
| Precision (no_helmet) | 85.9% |
| Recall (no_helmet) | 84.4% |
| F1-score (no_helmet) | 85.2% |
| Specificity (no_helmet) | 86.1% |

The Linear SVM achieved a reasonable balance between identifying cases where a helmet was missing and avoiding incorrect classifications.

### Comparison with Other Models

The project also compared results from different models developed by the group.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Linear SVM** | **85.3%** | 85.9% | 84.4% | 85.2% |
| Logistic Regression | 84.4% | 83.0% | 86.7% | 84.8% |
| K-Nearest Neighbours | 81.7% | 91.9% | 69.4% | 79.1% |
| Neural Network | 85.8% | 85.2% | 86.7% | 86.0% |

*Precision, recall and F1-score are reported for the `no_helmet` class.*

Although the Neural Network achieved the highest overall test accuracy, the difference compared with the Linear SVM was small.

The RBF SVM achieved **85.6% accuracy**, but the Linear SVM was retained because it provided better recall for the `no_helmet` class and was easier to interpret.

## My Contribution

This was a group project, with shared responsibilities for dataset preparation, preprocessing and feature extraction.

My individual work focused on the **Linear SVM classifier**, including:

- Preparing the shared feature datasets for modelling
- Handling missing and non-finite feature values
- Applying training-based feature scaling
- Implementing and tuning the Linear SVM
- Testing different cost parameters using cross-validation
- Comparing Linear and RBF SVM results
- Evaluating accuracy, precision, recall and F1-score
- Analysing classification errors and documenting findings

This helped me gain practical experience in machine learning experimentation, model validation, data preprocessing and technical reporting.

## Challenges and Limitations

### Dataset Consistency

The dataset combined images from different sources, including general human images and construction-related images.

This introduced differences in backgrounds, lighting, camera angles and image context.

Some predictions may therefore have been influenced by background characteristics rather than helmet presence alone.

### Model Performance

Although the Linear SVM achieved 85.3% accuracy, incorrect classifications remained.

In a safety-related application, missed `no_helmet` cases are particularly important because they represent potential violations that the model fails to identify.

### Generalisation

The results were obtained using one held-out test split.

Further testing using completely independent construction-site datasets would be necessary to assess how well the model generalises to real-world environments.

### Classification vs Detection

This project performs **image classification**, not object detection.

It predicts whether an image belongs to the `helmet` or `no_helmet` class. It does not locate individual workers, draw bounding boxes or track helmet compliance in real-time video.

The project should be considered an experimental decision-support prototype rather than a production-ready workplace safety system.

## Ethical Considerations

If a similar system were deployed using workplace photography or CCTV footage, important considerations would include:

- Data protection and privacy
- Appropriate handling and retention of images
- Model fairness across different environments
- Incorrect predictions and false negatives
- Transparency and human oversight

Automated predictions should not replace appropriate human safety checks.

## Future Improvements

Possible future improvements include:

- Expanding the dataset with more diverse construction-site images
- Testing generalisation on independent datasets
- Exploring convolutional neural networks and transfer learning
- Comparing classical feature extraction with deep learning approaches
- Improving handling of lighting, occlusion and camera-angle variation
- Investigating real-time helmet detection using object detection models
- Evaluating performance using additional validation strategies

## What I Learned

Through this project, I developed a better understanding of:

- Computer vision and image feature extraction
- Supervised machine learning classification
- Support Vector Machines and hyperparameter tuning
- Cross-validation and model selection
- Preventing data leakage during preprocessing
- Precision, recall and F1-score in safety-related classification
- Collaborative development and technical reporting

One of the biggest lessons was that selecting a machine learning model is not only about achieving the highest accuracy. It is also important to understand the type of errors the model makes and how those errors affect the intended application.

## Academic Context

**Project:** Helmet Compliance Classification in Construction-Site Images Using Machine Learning  
**Module:** Applied Machine Learning (CIS3036-N)  
**University:** Teesside University  
**Individual SVM Model and Report:** Deron Prince  
**Project Type:** University group project

This repository is intended to demonstrate academic machine learning development, experimentation and evaluation.
