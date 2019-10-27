from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics

from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd

import sys


def load_data(database_filepath):
    """
    Load the preprocessed data from the database file and split it into
    X, Y and category names datasets.

    :param database_filepath: Filepath of the database file
    :return: Pandas dataframes for the X, Y and category names
    """

    # Set up the SQL Alchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    df = pd.read_sql('disaster', engine)

    X = df[df.columns[:3]]
    Y = df[df.columns[3:]]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(df):
    """
    Apply the tokenization to the dataset.
    :param df: dataset of features
    :return: tokenized features
    """

    # Initialize a tokenizer that just looks for alphanumeric characters
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')

    # Use CountVectorizer to get the matrix of token counts
    # Ignore english stop words and use 1-grams only
    cv = CountVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range = (1,1),
        tokenizer = token.tokenize
    )

    # Apply the vectorizer to the message column and forget the rest
    text_counts = cv.fit_transform(df['message'])
    return text_counts


def build_model():

    return MultiOutputClassifier(MultinomialNB())

def evaluate_model(model, X_test, Y_test, category_names):


    Y_pred = model.predict(X_test)

    i = 0
    for category in category_names:

        y_test = Y_test[category]
        y_pred = Y_pred[:,i]
        print('MultinomialNB Accuracy for {}:'.format(category),
              metrics.accuracy_score(y_test, y_pred)
        )

        i += 1


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # Apply the tokenizer to our features
        X = tokenize(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()