import json
import plotly

from disaster_response_pipeline.models.train_classifier import load_data, lemmatized_words

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib


app = Flask(__name__)

# load data
X, Y, category_names = load_data('../data/disaster.db')

# load model
model = joblib.load("../models/disaster.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Most tagged categories
    top_categories = Y.sum().sort_values(0, ascending=False).head(5)
    top_category_counts = list(top_categories.values)
    top_category_names = list(top_categories.index)

    # Least tagged categories
    bot_categories = Y.sum().sort_values(0, ascending=True).head(5)
    bot_category_counts = list(bot_categories.values)
    bot_category_names = list(bot_categories.index)
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_category_names,
                    y=top_category_counts
                )
            ],

            'layout': {
                'title': 'Top 5 most tagged categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=bot_category_names,
                    y=bot_category_counts
                )
            ],

            'layout': {
                'title': 'Top 5 least tagged categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
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
    classification_results = dict(zip(Y, classification_labels))

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