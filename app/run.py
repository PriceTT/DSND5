import json
import plotly
import pandas as pd

import re

import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/disaster_model_Ad_Mod.sav")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    # Distribution of genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Mean of the features
    feat_means = df.loc[:, 'related':'direct_report'].mean()
    feat_names = list(df.loc[:, 'related':'direct_report'].columns)
    
    # Correlation of the top n features
    k = 10
    target_feature ='water'
    cols = abs(df.corr()).nlargest(k, target_feature)[target_feature]

    corr_names=list(cols.index)
    corr_values=cols.values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },


        
        {
            'data': [
                Heatmap(
                    z=df[corr_names].corr().values, 
                    x=corr_names,
                    y=corr_names,
                    colorscale='Viridis',
                )
            ],

            'layout': {
                'title': 'Top ten features correlated with the feature water',
                'height': 750,
                'margin': dict(
                    l = 150,
                    r = 30, 
                    b = 160,
                    t = 30,
                    pad = 4
                    ),
            }
        },
        
       {   'data': [
                Bar(
                    x=feat_names,
                    y=feat_means
                )
            ],

            'layout': {
                'title': 'Mean of the different features',
                'yaxis': {
                    'title': "Mean"
                },
                'xaxis': {
                    'title': "Features"
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()