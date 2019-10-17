from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

import pandas as pd
import sys

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(messages_filepath, categories_filepath):
    """
    Loads the messages and categories datasets from the local CSV files into pandas dataframes and merges them into a
    single pandas dataframe.

    :param messages_filepath: The filepath for the messages CSV file
    :param categories_filepath: The filepath for the categories CSV file
    :return: A merged pandas dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # set the indexes of both datasets as their 'id' columns in preparation for merging
    messages = messages.set_index('id')
    categories = categories.set_index('id')

    # merge datasets
    df = messages.merge(categories, left_index=True, right_index=True)

    return df


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

    # Instantiate the word stemmer
    lemmatizer = WordNetLemmatizer()

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
    return word_freqs, word_postags


def tokenize_column(text_col, top_n=200):
    """
    Apply the tokenizer to a column of text data

    :param text_col: column of text data
    :param top_n: (approximate) number of words to keep by top word frequency
    :return: two pandas dataframes of the tokenized output
    """

    df_freqs = []
    df_postags = []

    for text in text_col:
        word_freqs, word_postags = tokenize(text)
        df_freqs.append(word_freqs)
        df_postags.append(word_postags)

    # Convert to dataframes
    df_freqs = pd.DataFrame(df_freqs)
    df_postags = pd.DataFrame(df_postags)

    # Only keep top N words
    df_freqs = df_freqs[df_freqs.sum().nlargest(top_n).index.values]

    # Fill blank values with zero
    df_freqs = df_freqs.fillna(0)
    df_postags = df_postags.fillna(0)

    return df_freqs, df_postags


def prepare_features(df):
    """
    Apply feature engineering to the feature dataframe.

    :param df: Pandas dataframe containing the features
    :return: pre-processed features dataframe
    """

    # Get the word frequency and pos tag dataframes from the messages column
    word_freq, word_postags = tokenize_column(df['message'])

    print(word_freq.head())

    print('Calculating word TF-IDF...')
    # Create a matrix of TF-IDF for the word frequencies
    v_freq = TfidfVectorizer()
    df_freq = pd.DataFrame(v_freq.fit_transform(word_freq).toarray())
    df_freq.columns = ['freq_{}'.format(x) for x in v_freq.get_feature_names()]

    print('Calculating pos tags TF-IDF')
    # Create a matrix of TF-IDF for the word pos_tags
    v_postag = TfidfVectorizer()
    df_postag = pd.DataFrame(v_postag.fit_transform(word_postags).toarray())
    df_postag.columns = ['postag_{}'.format(x) for x in v_postag.get_feature_names()]

    # Append the new columns to the dataframe
    df = pd.concat([df_freq, df_postag, df], axis=1)

    # Drop the original columns
    df = df.drop(['message', 'original', 'genre'], axis=1)

    return df


def prepare_responses(df):
    """
    Apply transformations to the response variable

    :param df: dataframe of the unprocessed data
    :return: dataframe with the response variables parsed
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = [int(x.split('-')[1]) for x in categories[column]]

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True)

    return df


def clean_data(df):
    """
    Applies transformations to the disaster dataset to prepare it for the ML code
    :param df: A pandas dataframe of disaster responses
    :return: A cleaned pandas dataframe
    """

    # drop duplicates
    df = df.loc[~df.duplicated().values]

    print('Preparing response variables...')
    # prepare the response variables
    df = prepare_responses(df)

    print('Preparing features...')
    # prepare the features
    df = prepare_features(df)

    return df


def save_data(df, database_filename):
    """
    Saves the cleaned dataset into a local sqlite database file.

    :param df: The cleaned pandas dataframe
    :param database_filename: The filename of the database file
    :return:
    """

    # Set up the SQL Alchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filename))

    # Save the pandas dataframe as a sqlite database file
    df.to_sql('disaster', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()