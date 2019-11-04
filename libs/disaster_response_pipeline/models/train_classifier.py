from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import metrics

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.model_selection import train_test_split, GridSearchCV
from sqlalchemy import create_engine
import pandas as pd

import pickle

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

lemmatizer = WordNetLemmatizer()
analyzer = TfidfVectorizer().build_analyzer()

# Initialize a tokenizer that just looks for alphanumeric characters
token = RegexpTokenizer(r'[a-zA-Z0-9]+')

def lemmatized_words(doc):
    """
    Lemmatize the tokens contained within the document.
    :param doc: document of tokens
    :return: lemmatized document
    """
    return (lemmatizer.lemmatize(w) for w in analyzer(doc))


def build_model():
    """
    Initialize a model for running the data through
    :return: a model
    """

    # Set up the pipeline
    pipe =  Pipeline([
        ('vect', TfidfVectorizer(
            lowercase=True,
            tokenizer=token.tokenize,
            analyzer=lemmatized_words
        )),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])

    # Set up a parameter grid
    pg = [
        {
            'vect__ngram_range': [(1,1), (1,3)],
            'vect__stop_words': ['english', None]
        },
        {
            'clf': [MultiOutputClassifier(MultinomialNB())],
            'clf__estimator__alpha': [1.0, 0.3, 0.1, 0.01]
        },
        {
            'clf': [MultiOutputClassifier(DecisionTreeClassifier())],
            'clf__estimator__criterion': ['gini', 'entropy'],
            'clf__estimator__min_samples_split': [2, 5, 10]
        },
        {
            'clf': [MultiOutputClassifier(RandomForestClassifier())],
            'clf__estimator__n_estimators': [2,5, 10, 20],
            'clf__estimator__criterion': ['gini', 'entropy'],
            'clf__estimator__min_samples_split': [2, 5, 10]
        }
    ]

    return GridSearchCV(
        pipe, param_grid=pg, cv=3
    )



def tokenize(text, filepath='vectorizer.pkl'):
    """
    Apply the tokenization to the dataset.
    :param df: dataset of features
    :return: tokenized features
    """

    # Re-use the vectorizer from the given filepath
    v = pickle.load(open(filepath, 'rb'))

    # Apply the already fitted transform to the text
    text_counts = v.transform(text)

    return text_counts

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model results by printing the model scores for each category.
    :param model:
    :param X_test: The vectorized text features of the test data.
    :param Y_test: The response variables of the test data.
    :param category_names: The category names of the response variables.
    :return:
    """
    # Predict the output of the model on the test data
    Y_pred = model.predict(X_test)

    print('Individual variable evaluation:')

    i = 0
    # Run through each category and print its accuracy.
    for category in category_names:

        # The test values can be extracted using the category name since Y_test
        # is a pandas dataframe.
        y_test = Y_test[category]

        # The predicted values are in a numpy array so use indexing to extract
        # the predicted values.
        y_pred = Y_pred[:,i]

        # Print the accuracy score
        print('{}\n==============\nAccuracy: {}\nPrecision: {}\nRecall: {}\n==============\n'.format(
            category,
            metrics.accuracy_score(y_test, y_pred),
            metrics.precision_score(y_test, y_pred, average='micro'),
            metrics.recall_score(y_test, y_pred, average='micro')
        ))
        print(metrics.classification_report(y_test, y_pred))
        print(metrics.confusion_matrix(y_test, y_pred))

        i += 1

    print('Overall evaluation:')

    # Output the GridSearchCV best score and best params
    print('The best score from GridSearchCV: {}'.format(model.best_score_))
    print('Best model parameters:')
    print(model.best_params_)


def save_model(model, model_filepath):
    """
    Save the model as a pickle file to the filepath given.
    :param model: The model used to fit the data
    :param model_filepath: The filepath to save the model to
    :return:
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Processing function for the full sequence. Loads the data from the database,
    splits into train and test datasets, runs the model pipeline and fits the data
    to the model. Evaluation of the model is then performed and the model is output
    to a pickle file.
    :return:
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # We just want the message part
        X = X['message']

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