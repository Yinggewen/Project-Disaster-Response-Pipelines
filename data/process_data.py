import sys
import numpy as np
import pandas as pd
import nltk
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    loads the two datasets and merges both by the primary key,
    i.e. the id variable
    INPUT:
    messages_filepath - filepath for disaster_messages.csv
    categories_filepath - filepath for disaster_categories.csv
    OUTPUT:
    df - the merged dataframe
    .'''
    df1 = pd.read_csv(messages_filepath)
    df2 = pd.read_csv(categories_filepath)
    df = df1.merge(df2, on='id')
    return df


def clean_data(df):
    """ Cleans dataframe 
    
    Splits the categories column into separate, clearly named columns, 
    converts values to binary, 
    and drops duplicates.
    
    Args:
        df (dataframe): combined messages and categories dataframe.
        
    Returns:
        df (dataframe): clean dataframe.
        
    """
    categories = df.categories.str.split(';', expand=True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    categories.related.loc[categories.related=='related-2']='related-1'
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(subset='id', inplace=True)
    return df


def save_data(df, database_filename):
        '''
    Saves the data into a sqlite db file
    INPUT:
    df - the cleaned dataframe
    database_filename - where to store the database we create
    '''
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql("DisasterResponse", engine, index = False, if_exists='replace')
    
    

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