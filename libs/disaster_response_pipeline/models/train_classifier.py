from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import pandas as pd
import sys

lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    """
    Extracts the database data as a pandas dataframe
    :param database_filepath: The filepath to the database file
    :return: a pandas dataframe
    """

    # Connect and load data into pandas dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster', engine)

    # Split data into X, Y and the names of categories to classify
    X = df[df.columns[:3]]
    Y = df[df.columns[3:]]
    category_names = df.columns[3:]

    return X, Y, category_names


def tokenize(text):
    """
    Given a text value return a tokenized labelled output
    :param text: a string of free text
    :return: tokenized output
    """

    # Get word tokens from the text
    words = word_tokenize(text)

    # Get POS tags
    tagged = pos_tag(words)

    # Stem the words inside the POS tagged output
    stemtags = [(lemmatizer.lemmatize(x[0]), x[1]) for x in tagged]

    # Collect the POS tags and the word frequencies into separate dataframes
    word_freqs = {}
    word_postags = {}
    for x in stemtags:
        if x[0] in word_freqs:
            word_freqs[x[0]] += 1
        else:
            word_freqs[x[0]] = 1
        if x[1] in word_postags:
            word_postags[x[1]] += 1
        else:
            word_postags[x[1]] = 1

    # Return two pandas dataframes
    return pd.DataFrame([word_freqs]), pd.DataFrame([word_postags])


def tokenize_column(text_col):
    """
    Apply the tokenizer to a column of text data

    :param text_col: column of text data
    :return: two pandas dataframes of the tokenized output
    """

    df_freqs = pd.DataFrame()
    df_postags = pd.DataFrame()

    for text in text_col:
        word_freqs, word_postags = tokenize(text)
        df_freqs = pd.concat([df_freqs, word_freqs], sort=True)
        df_postags = pd.concat([df_postags, word_postags], sort=True)

    # Fill blank values with zero
    df_freqs = df_freqs.fillna(0)
    df_postags = df_postags.fillna(0)

    return df_freqs, df_postags

def prepare_data(df):
    """
    Apply feature engineering to the feature dataframe.

    :param df: Pandas dataframe containing the features
    :return: pre-processed features dataframe
    """

    # Get the word frequency and pos tag dataframes from the messages column
    word_freq, word_postags = tokenize_column(df['message'])

    # Create a matrix of TF-IDF for the word frequencies
    v_freq = TfidfVectorizer()
    df_freq = pd.DataFrame(v_freq.fit_transform(word_freq).toarray())
    df_freq.columns = ['freq_{}'.format(x) for x in v_freq.get_feature_names()]

    # Create a matrix of TF-IDF for the word pos_tags
    v_postag = TfidfVectorizer()
    df_postag = pd.DataFrame(v_postag.fit_transform(word_postags).toarray())
    df_postag.columns = ['postag_{}'.format(x) for x in v_postag.get_feature_names()]

    # Append the new columns to the dataframe
    df = pd.concat([df, f_freq, df_postag], axis=1)

    # Drop the original columns
    df = df.drop(['message', 'original', 'genre'], axis=1)

    return df

def build_model():
    pass

def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Preparing data')
        # Run feature engineering on the data
        X_train = prepare_data(X_train)
        X_test = prepare_data(X_test)

        print(X_train.head())
        print(X_test.head())
        
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