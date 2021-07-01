# Project-Disaster-Response-Pipelines


In this project, the data set contains real messages that were sent during disaster events. I will be creating a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

---

## Project Workspace - ETL
The first part of my data pipeline is the Extract, Transform, and Load process. Here, I will read the dataset, clean the data, and then store it in a SQLite database. 

---

## Project Workspace - Machine Learning Pipeline

* For the machine learning portion, I will split the data into a training set and a test set. 

* Create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). 

* Finally, export model to a pickle file. 
---

## The file structure of the project

In the last step, display results in a Flask web app. 
Here's the file structure of the project:

|--app <br>
|　　|-- template  <br>
|　　|-- |-- master.html  # main page of web app <br>
|　　|-- |-- go.html  # classification result page of web app <br>
|　　|-- run.py  # Flask file that runs app <br>
|　　| <br>
|--data <br>
|　　|-- disaster_categories.csv  # data to process <br>
|　　|-- disaster_messages.csv  # data to process <br>
|　　|-- process_data.py <br>
|　　|-- DisasterResponse.db   # database to save clean data to <br>
|　　|<br>
|--models <br>
|　　|-- train_classifier.py <br>
|　　|-- classifier.pkl  # saved model <br>
|　　|<br>
|--README.md <br>

---
## Running the Web App from the Project Workspace IDE

When working in the Project Workspace IDE, here is how to see your Flask app.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
