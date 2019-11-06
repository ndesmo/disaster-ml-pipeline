import pandas as pd
import sys

from sqlalchemy import create_engine


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


def clean_data(df):
    """
    Applies transformations to the disaster dataset to prepare it for the ML code
    :param df: A pandas dataframe of disaster responses
    :return: A cleaned pandas dataframe
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
        categories[column] = [min(int(x.split('-')[1]), 1) for x in categories[column]]

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.merge(categories, left_index=True, right_index=True)

    # drop duplicates
    df = df.loc[~df.duplicated().values]

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