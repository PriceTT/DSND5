import sys
# import libraries

import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

import pickle
import string
from sqlalchemy import create_engine

# Load ML libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer,classification_report
from sklearn.tree import DecisionTreeClassifier

def load_data(database_filepath):
    '''    Load 'messages' table from a database and extract X and Y values and category names.
    
    Parameters:
    database_filepath (string): path to SQLite db
    
    Returns:
    X (Dataframe):  feature DataFrame
    Y (Dataframe):  target DataFrame
    category_names (list):  used for data visualization (app)
    '''
    # load data from database
    db_location = 'sqlite:///' + database_filepath
    engine = create_engine(db_location)
    df = pd.read_sql_table('messages', engine)
    
    # Messages
    X = df['message']
    # Features
    Y = df.loc[:, 'related':'direct_report']
    # features col names
    col_names = list(Y.columns.values)
    return X, Y, col_names 


def tokenize(text):
    """Normalize, tokenize and stem text string
    
    Parameters:
    text (string): Single message as a string
       
    Returns:
    lemmed (list): List containing normalized and lemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    stop_words = stopwords.words("english")
    
    # lemmed word tokens and remove stop words
    lemmed = [WordNetLemmatizer().lemmatize(word, pos='v') for word in tokens if word not in stop_words]
    
    return lemmed


def build_model(X_train, Y_train):
    """ AdaBoostClassifier using Pipeline to process the text messages
    
    Parameters:
    X_train (array): features for training
    Y_train (array): target for training
    
    Returns:
    model (object): model for prediction
    """
    
    pipeline = Pipeline([
    ('Tdifvect', TfidfVectorizer(tokenizer = tokenize)),          #Bag of Words and Tf-idf
    ('clf', MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=300)))
    ])
    
    model = pipeline.fit(X_train, Y_train)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Return  f1 score, precision and recall for each output category of the dataset
    
    Parameters:
    model (object): model object to predict values
    X_test (array):  features for test
    Y_test (array):  target for test
    category_names (list): List containing names for each of the predicted fields.
    
    Returns:
    df (df): Dataframe f1 score, precision 
    """''
    
    # get actual and predicted values
    actual = np.array(Y_test)
    predicted = model.predict(X_test)   
        
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(category_names)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    df = pd.DataFrame(data = metrics, index = category_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return df


def save_model(model, model_filepath):
    """    This function saves trained model as Pickle file, to be loaded later.
    
    Parameters:
    model (object): GridSearchCV or Scikit Pipeline object
    model_filepath (string): destination path to save .pkl file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
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