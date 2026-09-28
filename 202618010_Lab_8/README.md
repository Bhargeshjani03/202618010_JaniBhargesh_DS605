Overview
This project implements traditional machine learning techniques for image and text classification using feature extraction and numerical representations.
The lab focuses on:
Extracting statistical and edge-based features from images.
Classifying asphalt images as Crack or Non-Crack.
Working with a pre-vectorized email spam dataset.
Comparing traditional machine learning classifiers.
Improving feature representation and analysing the trade-off between dimensionality, computation time, and classification performance.
Deep learning, CNNs, and pretrained image models are not used.
Part A — Image Feature Extraction and Classification
Dataset
The image dataset contains 400 asphalt images:
200 Crack images
200 Non-Crack images
The original images have dimensions of 448 × 448 × 3.
For computational efficiency, images were resized to 224 × 224 and converted to grayscale.
Image preprocessing
The preprocessing pipeline is:
Original Image
      ↓
Resize to 224 × 224
      ↓
Grayscale Conversion
      ↓
Feature Extraction
      ↓
Machine Learning Classification
Extracted Features
The following numerical features were extracted from each image:
Feature	Description
Mean Brightness	Average grayscale intensity
Contrast	Standard deviation of pixel intensities
Dark Pixel Ratio	Proportion of pixels below intensity 50
Bright Pixel Ratio	Proportion of pixels above intensity 200
Minimum Intensity	Minimum grayscale value
Maximum Intensity	Maximum grayscale value
Median Intensity	Median grayscale value
Edge Density	Proportion of pixels detected as edges
Canny edge detection was performed using:
Lower threshold = 100
Upper threshold = 200
The extracted feature table is available in:
outputs/image_features.csv
Image Classification Results
Baseline Logistic Regression
Accuracy	Precision	Recall	F1
93.75%	92.68%	95.00%	93.83%
Improved Logistic Regression
The Logistic Regression regularization parameter was tuned. The best result was obtained with:
C = 0.01
Accuracy	Precision	Recall	F1
97.78%	95.11%	97.33%	96.21%
The number of extracted features remained unchanged at 8.
The improvement therefore came from model hyperparameter tuning rather than increasing feature dimensionality.
Part B — Email Spam Classification
Dataset
The provided email dataset contains:
5,172 emails
3,000 word-frequency features
Prediction as the target variable
Class distribution:
Class	Count	Percentage
Non-Spam	3,672	70.99%
Spam	1,500	29.01%
The supplied CSV already contains numerical word-frequency features, so no additional TF-IDF transformation was applied.
Models
Two traditional classifiers were evaluated:
Multinomial Naive Bayes
Logistic Regression
Results
Model	Features	Accuracy	Precision	Recall	F1
Multinomial Naive Bayes	3,000	94.20%	86.81%	94.33%	90.42%
Logistic Regression	3,000	97.49%	93.35%	98.33%	95.78%
Logistic Regression achieved the higher classification performance on the provided word-frequency representation.
Part C — Improving Text Representation
To investigate the trade-off between dimensionality and performance, rare features were removed using a document-frequency threshold.
Features occurring in fewer than 20 training emails were removed.
This reduced the representation from:
3,000 → 2,574 features
This corresponds to approximately a 14.2% reduction in dimensionality.
Comparison
Representation	Features	Accuracy	F1	Training Time (s)	Prediction Time (s)
Original	3,000	97.49%	95.78%	0.1838	0.0124
Reduced	2,574	96.62%	94.27%	0.1813	0.0113
The reduced representation slightly decreased computation time and feature dimensionality, but classification performance also decreased.
This indicates that some relatively rare word features still contained useful information for distinguishing spam from legitimate emails.
Therefore, the original 3,000-feature representation was retained as the final text representation.
Key Observations
Simple statistical image features combined with Canny edge density were sufficient to achieve 97.78% accuracy with tuned Logistic Regression.
Logistic Regression performed better than Multinomial Naive Bayes for the provided email dataset.
The original text representation contained 3,000 features and achieved 97.49% accuracy.
Removing rare text features reduced dimensionality by 14.2%, but also reduced classification performance.
The experiments demonstrate that reducing dimensionality does not necessarily improve predictive performance because some low-frequency features may still contain useful information.
The image experiments also showed that maintaining the 224 × 224 resolution was preferable to aggressively reducing image resolution.
Technologies Used
Python
NumPy
Pandas
OpenCV
Matplotlib
Scikit-learn
Jupyter Notebook