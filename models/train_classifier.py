import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle

from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.svm import LinearSVC
from nltk.stem import WordNetLemmatizer


"""
def load_data(database_filepath):
    
   
    Load Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. disaster_response_db.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name

    engine = create_engine('sqlite:///' + database_filepath)
    #table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
    df = pd.read_sql_table(DisasterResponse,engine)
   
    X=df['message']
    y=df.iloc[:, 4:]
    category_names = y.columns
    df=df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]
    return X, y, category_names
"""

def load_data(database_filepath):
    """ Load data from database,
    Define feature and target variables X and y,
    Define categories list.
    
    Args:
        database_filepath (str): the database file path. 
        
    Returns: 
        df (dataframe): clean dataframe.
        X (dataframe): feature variable.
        y (dataframe): target variable.
        category_names (list): list of message categories.
    
    """
        
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine) 
    X = df['message']
    y = df.iloc[:,4:]
    category_names=y.columns 
    df=df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]
    
    return X, y, category_names


def tokenize(text):
    """using nltk to case normalize, lemmatize, and tokenize text. 
    This function is used in the machine learning pipeline to, 
    vectorize and then apply TF-IDF to the text.
    
     Args:
        text (str): A disaster message. 
        
    Returns: 
        processed_tokens (list): list of cleaned tokens in the message.
        
    """
    # get tokens from text
    tokens= WhitespaceTokenizer().tokenize(text)
    lemmatizer= WordNetLemmatizer()
    
    # clean tokens
    processed_tokens=[]
    for token in tokens:
        token=lemmatizer.lemmatize(token).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        token=re.sub(r'\[[^.,;:]]*\]','', token)
        
        # add token to compiled list if not empty
        if token !='':
            processed_tokens.append(token)
    return processed_tokens





def build_model():
    """
    Build Pipeline function
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC())))
    ])
    
    
    parameters = {
    'tfidf__smooth_idf':[True, False],
    'clf__estimator__estimator__C':[1,2,5]}
    
    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy and f1score)
    
    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    Y_pred = model.predict(X_test)
    
    # multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average overall accuracy {0:.2f}%'.format(overall_accuracy*100))
    #print('F1 score (custom definition) {0:.2f}%'.format(multi_f1*100))

    # Print the whole classification report.
    Y_pred = pd.DataFrame(Y_pred, columns = Y_test.columns)
    
    for column in Y_test.columns:
        print('Model Performance with Category: {}'.format(column))
        print(classification_report(Y_test[column],Y_pred[column]))


        
def save_model(model, model_filepath):
    """
    Save Pipeline function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        pipeline -> GridSearchCV or Scikit Pipelin object
        pickle_filepath -> destination path to save .pkl file
    
    """
    pickle.dump(model, open(model_filepath, 'wb'))
        
        


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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