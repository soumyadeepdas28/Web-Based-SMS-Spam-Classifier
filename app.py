# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from os import environ

app = Flask(__name__)

# Read in data set
data = pd.read_csv('mail_data.csv')



#missing values
data.isnull().sum()

 #check for duplicate values
data.duplicated().sum()

data = data.drop_duplicates(keep='first')


# Preprocess the data
data['Message'] = data['Message'].str.lower() # Convert text to lowercase
data['Message'] = data['Message'].str.replace('[^\w\s]','') # Remove punctuation

# Convert text to numerical feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])
y = data['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)

@app.route('/')
def index():
    return render_template('index.html')


# Predict label of new SMS message
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        new_sms = request.form['message']
    try:
            
            
        # Preprocess the input message
        new_sms = vectorizer.transform([new_sms])
        # Make a prediction
        prediction = clf.predict(new_sms)
    except:
            return ('error')
            

    # Print prediction
    if prediction[0] == 'spam':
        return render_template(
        'spam.html')
    else:
        return render_template(
        'not spam.html')

if __name__ == '__main__':
        HOST = environ.get('SERVER_HOST', 'localhost')
        try:
            PORT = int(environ.get('SERVER_PORT', '5555'))
        except ValueError:
            PORT = 5555
        app.run(HOST, PORT)
