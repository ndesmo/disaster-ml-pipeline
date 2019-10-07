import sys

import pandas as pd


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
    pass


def save_data(df, database_filename):
    pass  


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