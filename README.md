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



