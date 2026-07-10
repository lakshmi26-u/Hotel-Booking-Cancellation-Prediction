# Hotel Booking Cancellation Prediction

## Project Overview

This project was completed as part of the Applied AI & ML Essentials Capstone Project.
The aim of this project is to build a machine learning model that predicts whether a hotel booking will be cancelled or not. The project follows the complete AI and Machine Learning lifecycle from data cleaning to model development and LLM integration.

### Project Parts

- Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis
- Part 2 – Supervised Machine Learning
- Part 3 – Advanced Machine Learning and Model Optimization
- Part 4 – LLM Integration and AI Chatbot

---

## Dataset

**Dataset Name:** Hotel Booking Demand Dataset

**Target Variable:** `is_canceled`

This dataset contains hotel booking information such as booking details, customer information, room type, hotel type, lead time, ADR, previous cancellations and many other features that help predict whether a booking will be cancelled.

## Part 1 – Data Acquisition, Cleaning and Exploratory Data Analysis

### What I Did

In Part 1, I cleaned the raw hotel booking dataset and explored it using different data analysis techniques before building any machine learning models.

The following tasks were completed:

- Loaded the dataset using pandas.
- Checked the shape, data types and first five rows.
- Identified missing values and calculated their percentages.
- Filled missing numeric values with the median where appropriate.
- Found and removed duplicate records.
- Converted suitable columns to correct data types.
- Generated descriptive statistics.
- Calculated skewness for numeric columns.
- Detected outliers using the IQR method.
- Created different visualizations for better understanding of the data.
- Compared Pearson and Spearman correlation.
- Performed grouped aggregation.
- Saved the cleaned dataset as `cleaned_data.csv`.

## Key Findings

- Most hotel bookings were not cancelled.
- City Hotels had more bookings than Resort Hotels.
- Customers with longer lead times were more likely to cancel their bookings.
- Most bookings were made without a deposit.
- Online TA was the largest market segment.
- The hotel dataset contained some outliers, but they were kept for model training in the next part.
- The cleaned dataset was saved as `cleaned_data.csv` for use in Part 2 and Part 3.

## Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Files

- Part_1_Hotel_Booking_Data_Cleaning_and_EDA.ipynb
- cleaned_data.csv
- README.md

## Part 1 Conclusion

In Part 1, I cleaned and explored the hotel booking dataset before building any machine learning models. I handled missing values, removed duplicates, corrected data types, detected outliers, and created different visualizations to understand the data. The cleaned dataset was saved as `cleaned_data.csv`, which will be used in Part 2 and Part 3.
The remaining parts of this project will focus on building machine learning models, improving model performance, and developing an AI chatbot that predicts hotel booking cancellations using the trained model.

---

# Part 2 – Supervised Machine Learning

## Objective

The objective of Part 2 was to build and evaluate machine learning models using the cleaned hotel booking dataset. I trained regression and classification models, evaluated their performance using different metrics, and compared multiple approaches to understand which model performed better for the prediction tasks.

## Feature and Target Selection

For this project, I used the cleaned dataset generated in Part 1.

### Feature Matrix (X)

The feature matrix contains all the input variables except the target columns. Before training the models, I removed the following columns:

- adr (used as the regression target)
- is_canceled (used as the classification target)
- reservation_status
- reservation_status_date

### Target Variables

Regression Target (y_reg)

- adr (Average Daily Rate)

Classification Target (y_clf)

- is_canceled
  - 0 = Booking Not Cancelled
  - 1 = Booking Cancelled

## Data Preprocessing

Before training the machine learning models, I performed the following preprocessing steps:

- Encoded all categorical features using one-hot encoding.
- Used `drop_first=True` to avoid multicollinearity.
- Split the dataset into 80% training data and 20% testing data.
- Used `random_state=42` to make the results reproducible.
- Applied StandardScaler to standardize the feature values.
- The scaler was fitted only on the training data and then applied to the test data to prevent data leakage.

### Why One-Hot Encoding?

Most categorical columns in this dataset, such as hotel type, country, meal, and market segment, do not have any natural order. One-hot encoding converts these categories into separate binary columns without creating a false numerical relationship between them.

### Why StandardScaler?

The numerical features have different ranges. StandardScaler transforms them to a similar scale, which helps regression and logistic regression models learn more effectively.

### Preventing Data Leakage

The StandardScaler was fitted only on the training dataset. The same fitted scaler was then used to transform the test dataset. This prevents information from the test data from influencing the training process, resulting in a fair model evaluation.

## Linear Regression

Linear Regression was used to predict the Average Daily Rate (ADR) of a hotel booking.

### Model Performance

- Mean Squared Error (MSE): **985.5543**
- R² Score: **0.6476**

The model was able to explain about **64.76%** of the variation in ADR, which indicates a reasonably good fit for this dataset.

### Most Important Features

The three features with the largest absolute coefficients were:

- arrival_date_month_August
- arrival_date_week_number
- arrival_date_month_July

### Coefficient Interpretation

A positive coefficient means that when the value of that feature increases by one standardized unit, the predicted ADR also increases, while keeping the other features constant.

A negative coefficient means that when the value of that feature increases by one standardized unit, the predicted ADR decreases, while keeping the other features constant.

## Ridge Regression

To compare the performance of Linear Regression, I also trained a Ridge Regression model with **alpha = 1.0**.

### Model Comparison

| Model | MSE | R² Score |
|------|------:|------:|
| Linear Regression | 985.5543 | 0.6476 |
| Ridge Regression | 985.5897 | 0.6476 |

### Interpretation

The Ridge Regression model produced results that were very similar to the Linear Regression model.

Ridge Regression applies L2 regularization, which helps reduce the size of the model coefficients and can reduce overfitting. The **alpha** parameter controls the strength of this regularization. A larger alpha value applies stronger regularization, while a smaller value behaves more like ordinary Linear Regression.

## Logistic Regression

Logistic Regression was used to predict whether a hotel booking would be cancelled or not.

### Class Imbalance

Before training the model, I checked the distribution of the target classes.

- Not Cancelled (0): 72.41%
- Cancelled (1): 27.59%

Since the cancelled bookings represented less than 35% of the training data, I used `class_weight='balanced'` in the Logistic Regression model. This gives more importance to the minority class during training without changing the original dataset.

### Model Performance

- Accuracy: **74.78%**
- Precision: **52.38%**
- Recall: **77.71%**
- F1-Score: **62.58%**
- ROC-AUC Score: **0.8390**

### Confusion Matrix

| Actual | Predicted Not Cancelled | Predicted Cancelled |
|---------|------------------------:|--------------------:|
| Not Cancelled | 9382 | 3350 |
| Cancelled | 1057 | 3685 |

## ROC Curve and AUC

The ROC (Receiver Operating Characteristic) curve was plotted to evaluate the classification model at different decision thresholds.

The Logistic Regression model achieved an **AUC score of 0.8390**.

An AUC score close to 1 indicates that the model can effectively distinguish between cancelled and non-cancelled bookings. An AUC of 0.8390 shows that the model has good discrimination ability and performs well on this classification task.

## Precision and Recall

### Precision Formula

Precision = TP / (TP + FP)

Precision measures how many bookings predicted as cancelled were actually cancelled.

### Recall Formula

Recall = TP / (TP + FN)

Recall measures how many of the actual cancelled bookings were correctly identified by the model.

### Which Metric is More Important?

For hotel booking cancellation prediction, **Recall** is more important than Precision.

Missing a booking that will actually be cancelled (a false negative) may cause the hotel to lose the opportunity to resell the room or plan resources effectively. Predicting some extra cancellations (false positives) is generally less costly than missing real cancellations.

Therefore, improving Recall is more beneficial for this business problem.

## Decision Threshold Analysis

The Logistic Regression model was evaluated using different decision thresholds to understand how Precision, Recall, and F1-score change.

| Threshold | Precision | Recall | F1-Score |
|----------:|----------:|--------:|---------:|
| 0.30 | 0.4198 | 0.9323 | 0.5789 |
| 0.40 | 0.4703 | 0.8703 | 0.6106 |
| 0.50 | 0.5238 | 0.7771 | 0.6258 |
| 0.60 | 0.5853 | 0.6598 | 0.6203 |
| 0.70 | 0.6449 | 0.5156 | 0.5731 |

### Interpretation

The highest F1-score (**0.6258**) was obtained at the default threshold of **0.50**.

If the goal is to identify as many cancelled bookings as possible, the threshold can be lowered. This increases Recall but also increases the number of false positives.

If the goal is to reduce false alarms, the threshold can be increased. This improves Precision but may miss more actual cancelled bookings.

For this project, the default threshold of **0.50** provides the best balance between Precision and Recall.
## Logistic Regression Regularization Experiment

To understand the effect of regularization, I trained a second Logistic Regression model using **C = 0.01** and compared it with the baseline model (**C = 1.0**).

### Model Comparison

| Model | Precision | Recall | AUC |
|------|----------:|-------:|----:|
| C = 1.0 | 0.5238 | 0.7771 | 0.8390 |
| C = 0.01 | 0.5225 | 0.7727 | 0.8385 |

### Interpretation

The model with **C = 0.01** produced performance that was very similar to the baseline model, but its Precision, Recall, and AUC were slightly lower.

The **C** parameter controls the amount of regularization in Logistic Regression. A smaller value of **C** applies stronger regularization, which reduces the size of the model coefficients and helps prevent overfitting. A larger value allows the model to fit the training data more closely.

In this project, reducing **C** to **0.01** did not improve the model performance, so the baseline model (**C = 1.0**) remained the better choice.
## Bootstrap Confidence Interval for AUC Difference

To check whether the difference between the two Logistic Regression models was reliable, I performed a bootstrap analysis using **500 bootstrap samples**.

### Results

- Mean AUC Difference: **0.000542**
- 95% Confidence Interval:
  - Lower Bound: **0.000239**
  - Upper Bound: **0.000853**

### Interpretation

The 95% confidence interval does **not** include zero. This indicates that the baseline Logistic Regression model (C = 1.0) consistently performed slightly better than the model with stronger regularization (C = 0.01) across different bootstrap samples.

Although the improvement is small, the results suggest that the baseline model provides more reliable performance on this dataset.
## Part 2 Conclusion

In Part 2, I built and evaluated both regression and classification machine learning models using the cleaned dataset from Part 1.

For regression, I trained Linear Regression and Ridge Regression models to predict the Average Daily Rate (ADR). Their performance was compared using Mean Squared Error (MSE) and R² score.

For classification, I trained a Logistic Regression model to predict whether a hotel booking would be cancelled. The model was evaluated using a confusion matrix, accuracy, precision, recall, F1-score, ROC curve, and AUC score. I also analyzed different decision thresholds, compared two levels of regularization (C = 1.0 and C = 0.01), and used bootstrap sampling to evaluate the reliability of the AUC difference between the models.

The results show that the baseline Logistic Regression model (C = 1.0) provided the best overall performance and will be used as a baseline for the advanced machine learning models in Part 3.

Completed Part 2 - Supervised Machine Learning

