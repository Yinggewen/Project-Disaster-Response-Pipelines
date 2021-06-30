# Project-Disaster-Response-Pipelines


In this project, the data set contains real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 


## Project Workspace - ETL
The first part of my data pipeline is the Extract, Transform, and Load process. Here, I will read the dataset, clean the data, and then store it in a SQLite database. 


## Project Workspace - Machine Learning Pipeline

* For the machine learning portion, I will split the data into a training set and a test set. 

* Create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). 

* Finally, export model to a pickle file. 

## Flask App

In the last step, display results in a Flask web app. 
Here's the file structure of the project:

|-app

  | - template
  | |- master.html  # main page of web app
  | |- go.html  # classification result page of web app
  |- run.py  # Flask file that runs app

|-data
  |- disaster_categories.csv  # data to process 
  |- disaster_messages.csv  # data to process
  |- process_data.py
  |- DisasterResponse.db   # database to save clean data to

|-models
  |- train_classifier.py
  |- classifier.pkl  # saved model 

- README.md
