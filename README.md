# Sentiment Analysis on IMDB Movie Reviews

## Overview

This project demonstrates how to perform sentiment analysis on movie reviews from the IMDB dataset. Sentiment analysis is a natural language processing (NLP) technique used to determine whether a piece of text is positive, negative, or neutral. In this project, we classify IMDB movie reviews as either positive or negative.

## Dataset

The dataset used is the IMDB movie reviews dataset, which contains 50,000 reviews split evenly into 25,000 for training and 25,000 for testing. Each review is labeled as either positive (1) or negative (0).

## Project Structure

- **notebooks/**
  - `Sentiment_Analysis_IMDB.ipynb`: The Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and predictions.
  
- **data/**
  - `imdb_reviews_train.csv`: Preprocessed training dataset.
  - `imdb_reviews_test.csv`: Preprocessed testing dataset.

- **models/**
  - `sentiment_model.pkl`: The trained sentiment analysis model (if saved).

- **README.md**: This file, providing an overview of the project.

## Installation

To run this project, you need Python installed with the following libraries:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn tensorflow
```

## Usage

1. **Data Loading**: Load the IMDB dataset using TensorFlow's built-in `datasets` module.

2. **Data Preprocessing**: 
   - Convert integer sequences back into text using the word index.
   - Remove stopwords, punctuation, and perform other preprocessing steps.

3. **Model Building**:
   - Use a `Pipeline` with `TfidfVectorizer` and `MultinomialNB` for text vectorization and sentiment classification.
   
4. **Model Training**:
   - Train the model on the training dataset.

5. **Evaluation**:
   - Evaluate the model on the testing dataset and print metrics like accuracy, precision, recall, F1-score, and confusion matrix.

6. **Prediction**:
   - Use the trained model to predict the sentiment of new movie reviews.

## Results

The model achieved an accuracy of **84.6%** on the test set. Below are the detailed metrics:

- **Accuracy**: 0.84632
- **Precision**: 0.84 (Negative), 0.86 (Positive)
- **Recall**: 0.86 (Negative), 0.83 (Positive)
- **F1-Score**: 0.85 (Negative), 0.84 (Positive)

## Example Usage

```python
# Load the model
import pickle
with open('models/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict sentiment for new reviews
new_reviews = ["This movie was fantastic! I loved the plot and the acting was superb.",
               "Terrible movie. It was a waste of time, and I hated the script."]
predictions = model.predict(new_reviews)
print(predictions)  # Output: [1, 0]
```

## Next Steps

- **Model Tuning**: Explore different models or tweak hyperparameters to improve performance.
- **Feature Engineering**: Experiment with different text preprocessing techniques.
- **Deployment**: Deploy the model as a web service or integrate it into an application.

## Contributing

Feel free to fork this project, make improvements, and submit pull requests. All contributions are welcome!

## Contact

For any questions or inquiries, please contact [Devaloy Mukherjee](mailto:devaloy.mukherjee@gmail.com).

**Note**: This readme file was generated using chatgpt3.5 turbo instruct api
