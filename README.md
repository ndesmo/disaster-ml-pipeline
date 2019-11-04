# disaster-ml-pipeline
Classifying free text responses to a disaster

In this project I am providing code that consumes messages from various disaster events, performs ETL and fits a model to classify the data into any of 36 categories.

An app provided in the app directory allows the user to provide free text input and see the categories that the model classifies the text into.

In order for the app to work, please first run the data/process_data.py file, followed by the models/train_classifier.py file.

## Libraries required

It is recommended to use Anaconda, so that the following libraries will be simple to install or already installed:

* pandas
* sklearn
* nltk
* sqlalchemy
* plotly
* flask

Also, the code refers to itself. Please add the disaster_response_pipeline directory to your PYTHONPATH.

## License

This project is published under the Apache 2.0 Open Source License
http://www.apache.org/licenses/LICENSE-2.0

## Acknowledegments
This project is submitted as part of the Data Scientist Nanodegree at Udacity, which I would highly recommend.