# DS605: Fundamentals of Machine Learning

## Lab Assignment 4 — End-to-End Machine Learning: Airbnb Price Prediction

### Project Overview

This project implements an end-to-end machine learning workflow for predicting the **nightly price of Airbnb listings in New York City** using the Kaggle **New York City Airbnb Open Data (AB_NYC_2019.csv)** dataset.

The project covers the complete machine learning pipeline, including:

* Data analysis and exploration
* Data cleaning and preprocessing
* Feature engineering
* Feature selection
* Regression model training and comparison
* Hyperparameter tuning
* Final model evaluation
* Model and preprocessing artifact saving
* Streamlit application development
* Price prediction for new Airbnb listings

The final trained model is integrated into a **Streamlit web application** that allows users to enter listing information and receive an estimated nightly Airbnb price.

---

## Objectives

The main objectives of this project are:

1. Analyze and prepare the Airbnb dataset for machine learning.
2. Identify important factors affecting Airbnb listing prices.
3. Train and compare suitable regression models.
4. Tune and select an appropriate final model.
5. Evaluate the final model using regression metrics.
6. Save the trained model and preprocessing artifacts.
7. Develop a Streamlit application for making predictions on new listings.
8. Provide an end-to-end, reproducible machine learning workflow.

---

## Dataset

**Dataset:** New York City Airbnb Open Data

**File:** `AB_NYC_2019.csv`

The dataset contains Airbnb listings from New York City and includes information such as:

* Listing name
* Host information
* Neighbourhood
* Neighbourhood group
* Latitude and longitude
* Room type
* Price
* Minimum nights
* Number of reviews
* Reviews per month
* Host listing count
* Availability throughout the year

The target variable for this project is:

```text
price
```

which represents the nightly price of an Airbnb listing.

---

# Project Structure

```text
202618010_Lab_5/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   └── AB_NYC_2019.csv
│
├── models/
│   ├── airbnb_xgb_118.pkl
│   ├── airbnb_preprocessor.pkl
│   ├── airbnb_tfidf_100.pkl
│   ├── neighbourhood_room_type_te.pkl
│   └── te_global_mean.pkl
│
└── notebooks/
    └── 202618010_Assignment_4(4).ipynb
```

> The exact folder name and notebook location can be adjusted according to the final GitHub repository structure.

---

# Task 1 — Data Analysis and Preparation

## 1. Exploratory Data Analysis

The dataset was analyzed to understand:

* Distribution of Airbnb prices
* Distribution of listings across neighbourhood groups
* Effect of room type on price
* Geographic distribution of listings
* Relationship between reviews and price
* Availability patterns
* Host listing patterns
* Missing values and potential outliers

The analysis was used to identify variables that could contribute to predicting Airbnb prices.

---

## 2. Data Cleaning

The dataset was checked for missing values and unsuitable values.

Appropriate preprocessing and transformations were applied before model training.

Price outliers were also handled using a capped target value so that extremely expensive listings would not disproportionately influence model training.

---

## 3. Feature Engineering

Several transformations were applied to improve model performance.

Log transformations were used for highly skewed variables such as:

```text
minimum_nights
number_of_reviews
reviews_per_month
calculated_host_listings_count
```

The transformed features used by the final model include:

```text
log_minimum_nights
log_number_of_reviews
log_reviews_per_month
log_host_listings
```

A geographic feature was also derived:

```text
distance_to_manhattan
```

This represents the approximate distance of a listing from Manhattan.

---

## 4. Target Encoding

Target encoding was used for selected categorical variables.

The final feature set includes target-encoded representations of:

```text
neighbourhood
neighbourhood_group
room_type
```

An interaction-based target encoding was also used for:

```text
neighbourhood × room_type
```

This allows the model to capture differences in prices associated with combinations of neighbourhood and room type.

---

## 5. Text Feature Engineering

The Airbnb listing `name` was used as a text feature.

TF-IDF vectorization was applied to listing names, producing **100 TF-IDF features**.

This allows the model to capture useful information from words appearing in listing names.

---

# Final Feature Set

The final model uses **118 features**.

The feature representation consists of:

* Structured/preprocessed listing features
* Target-encoded categorical information
* 100 TF-IDF features derived from listing names

Important structured variables include:

```text
latitude
longitude
availability_365
log_minimum_nights
log_number_of_reviews
log_reviews_per_month
log_host_listings
neighbourhood target encoding
neighbourhood_group target encoding
room_type target encoding
```

---

# Task 2 — Model Training and Evaluation

## Model Development

Multiple regression approaches were considered during model development.

The final model selected for the prediction application is an:

### XGBoost Regression Model

XGBoost was selected because it can effectively model nonlinear relationships and interactions between the different Airbnb listing features.

The final trained model uses the engineered **118-feature representation**.

---

## Final Model Performance

The final XGBoost model achieved the following results:

| Metric | Training Set | Test Set |
| ------ | -----------: | -------: |
| MAE    |        33.89 |    47.61 |
| RMSE   |        59.33 |    86.76 |
| R²     |       0.7634 |   0.4926 |

### Interpretation

The test-set **MAE of approximately $47.61** means that the model's predicted Airbnb price differs from the actual price by about $47.61 on average.

The test-set **R² of approximately 0.493** indicates that the model explains around 49.3% of the variation in the target price on unseen test data.

The difference between training and test performance indicates some degree of overfitting, although the model still provides useful predictive performance.

---

# Saved Model and Preprocessing Artifacts

The trained model and preprocessing components are saved so that the same workflow can be used when making predictions on new Airbnb listings.

The following files are included:

```text
models/
├── airbnb_xgb_118.pkl
├── airbnb_preprocessor.pkl
├── airbnb_tfidf_100.pkl
├── neighbourhood_room_type_te.pkl
└── te_global_mean.pkl
```

### Artifact Description

| File                             | Purpose                                   |
| -------------------------------- | ----------------------------------------- |
| `airbnb_xgb_118.pkl`             | Final trained XGBoost model               |
| `airbnb_preprocessor.pkl`        | Preprocessing pipeline                    |
| `airbnb_tfidf_100.pkl`           | TF-IDF vectorizer for listing names       |
| `neighbourhood_room_type_te.pkl` | Neighbourhood × room type target encoding |
| `te_global_mean.pkl`             | Global mean used for target encoding      |

These artifacts allow the Streamlit application to apply the same transformations used during model training.

---

# Task 3 — Streamlit Application

A Streamlit application was developed to make the trained machine learning model usable through a simple web interface.

The application is implemented in:

```text
app.py
```

The user can enter relevant Airbnb listing information such as:

* Neighbourhood group
* Neighbourhood
* Latitude
* Longitude
* Room type
* Minimum nights
* Number of reviews
* Reviews per month
* Host listing count
* Availability
* Listing name

The application processes these inputs using the saved preprocessing and feature-engineering components and passes the resulting 118-feature representation to the trained XGBoost model.

The predicted nightly price is then displayed to the user.

---

## Example Prediction

For a sample listing with characteristics such as:

```text
Neighbourhood Group: Manhattan
Neighbourhood: Upper West Side
Latitude: 40.7850
Longitude: -73.9750
Room Type: Entire home/apt
Minimum Nights: 3
Number of Reviews: 50
Reviews per Month: 2.0
Host Listings Count: 1
Availability: 200
Listing Name: Beautiful cozy apartment near Central Park
```

the application produces an estimated nightly price of approximately:

```text
$203.65
```

---

# Running the Application Locally

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

## 2. Install Dependencies

Create and activate a virtual environment if desired, then run:

```bash
pip install -r requirements.txt
```

## 3. Start the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# Requirements

The main Python libraries used in this project include:

```text
pandas
numpy
scikit-learn
xgboost
streamlit
joblib
scipy
matplotlib
seaborn
```

The complete dependency list is provided in:

```text
requirements.txt
```

---

# Task 4 — Final Project Summary

## Key Findings

The analysis indicates that Airbnb prices are influenced by several factors, including:

* Location
* Neighbourhood
* Room type
* Geographic coordinates
* Minimum stay requirements
* Listing and host characteristics
* Availability
* Review-related variables
* Information contained in the listing name

Location and room type are particularly important because Airbnb prices can vary substantially across different parts of New York City and between different types of accommodation.

---

## Final Model

The final selected model is an **XGBoost regression model** using a combination of:

1. Structured numerical features
2. Target-encoded categorical features
3. Neighbourhood × room-type interaction encoding
4. TF-IDF features from listing names

The resulting feature representation contains **118 features**.

---

## Application Result

The trained model was successfully integrated into a Streamlit application.

The application provides an easy-to-use interface where a user can enter listing characteristics and obtain an estimated nightly Airbnb price without directly interacting with the machine learning code.

---

# Limitations

The developed system has several limitations:

* The dataset represents Airbnb listings from **2019**, so it may not accurately reflect current Airbnb prices.
* Airbnb prices are affected by factors that may not be available in the dataset.
* The model does not capture all temporal effects such as holidays, special events, or seasonal demand.
* The model has a test R² of approximately 0.49, meaning a substantial portion of price variation remains unexplained.
* The difference between training and test performance indicates some overfitting.
* Predictions should therefore be treated as estimates rather than exact market prices.
* The model should not be interpreted as a guarantee of the actual price a listing can obtain.

---

# GitHub Repository

**Repository:**
`<ADD YOUR PUBLIC GITHUB REPOSITORY LINK HERE>`

# Deployed Application

**Streamlit Application:**
`<ADD YOUR DEPLOYED STREAMLIT LINK HERE>`

If the application is not deployed, this section can be removed or replaced with:

```text
The application can be run locally using the instructions provided above.
```

---

# Conclusion

This project demonstrates a complete end-to-end machine learning workflow for Airbnb price prediction, starting from raw data analysis and preprocessing and continuing through feature engineering, model training, evaluation, artifact saving, and deployment through a Streamlit application.

The final XGBoost model provides a practical way to estimate Airbnb nightly prices from listing characteristics while maintaining a reproducible preprocessing and prediction pipeline.

---

## Author

**Name:** Bhargesh
**Course:** DS605 — Fundamentals of Machine Learning
**Assignment:** Lab Assignment 4 — End-to-End Machine Learning: Airbnb Price Prediction
