# Fake News Detection using Logistic Regression

## Overview
This project focuses on detecting fake news articles using a Logistic Regression model. The model processes textual data from news articles (title and author) to classify them as "Real" or "Fake." The workflow includes text preprocessing, feature extraction with TF-IDF, model training, and evaluation. The final model achieves **98% accuracy** on test data.

## What’s Inside the Dataset?
The dataset contains **20,800 news articles**, each with the following details:
- `id`: A unique identifier for the article.
- `title`: The headline of the article.
- `author`: The person or organization who wrote the article.
- `text`: The main content of the article (sometimes incomplete).
- `label`: A binary label (`1` for fake news, `0` for real news).

## How Does It Work?
The project follows a clear, step-by-step process:
1. **Preprocessing**:
   - Missing values in the dataset are replaced with empty strings.
   - The `author` and `title` columns are combined into a single `content` column for analysis.
   - The text is cleaned by removing non-alphabetic characters, converting to lowercase, and breaking it into individual words (tokenization).
   - Stopwords (like "the," "and," etc.) are removed, and words are reduced to their root form (stemming).

2. **Feature Extraction**:
   - The cleaned text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**, a technique that highlights the importance of words in the context of the dataset.

3. **Model Training**:
   - The dataset is split into training (80%) and testing (20%) sets.
   - A **Logistic Regression** model is trained on the training data.

4. **Evaluation**:
   - The model’s performance is evaluated on both the training and test datasets, achieving **98.66%** and **97.91%** accuracy, respectively.

5. **Prediction**:
   - You can use the trained model to predict whether a new article is real or fake.

## How to Get Started
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
   
2. **Install Dependencies**:
Make sure you have the required libraries installed:
   ```bash
   pip install numpy pandas nltk scikit-learn

3. **Download NLTK Stopwords**:
Run the following Python code to download the stopwords dataset:
   ```bash
   import nltk
   nltk.download('stopwords')

4. **Add the Dataset**:
Place the train.csv file in the project directory. You can find the dataset on Kaggle or use your own.

Run the Notebook:
Open the Jupyter Notebook (ML Project.ipynb) and follow the steps to preprocess the data, train the model, and make predictions.

## Results
   Training Accuracy: 98.66%
   Test Accuracy: 97.91%
