from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Ensure required NLTK data packages are available
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv('kaggle_parsed_dataset.csv')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    text = re.sub(r'\d+', '', text)
    return text

# Clean the dataset
df_cleaned = df.dropna(subset=['Text', 'oh_label'])
df_cleaned['Text'] = df_cleaned['Text'].apply(preprocess_text)

# Split the data into features and labels
X = df_cleaned['Text']
y = df_cleaned['oh_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Naive Bayes': MultinomialNB()
}

# Initialize dictionaries to store accuracies
training_accuracies = {}
testing_accuracies = {}

# Train and evaluate the models, plot accuracies, and save the best model
accuracy_scores = {}
best_model_name = ''
best_accuracy = 0

for model_name, model in models.items():
    # Fit model to the training data
    model.fit(X_train_tfidf, y_train)
    
    # Predict on training and testing data
    y_train_pred = model.predict(X_train_tfidf)
    y_test_pred = model.predict(X_test_tfidf)
    
    # Calculate training and testing accuracy
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Store accuracies in dictionaries
    training_accuracies[model_name] = train_accuracy
    testing_accuracies[model_name] = test_accuracy
    
    # Save accuracy for the model selection process
    accuracy_scores[model_name] = test_accuracy
    
    # Check for best model based on test accuracy
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model_name = model_name
        best_model = model

# Print best model details
# Output the best model and accuracy
print("Model Accuracies:")
for model_name, accuracy in accuracy_scores.items():
    print(f"{model_name} Accuracy: {accuracy * 100:.2f}%")
print(f"Best Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%")


# Save the best model and the vectorizer
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Plotting the accuracies as a bar graph
def plot_training_testing_accuracies():
    # Bar width and positions for the bar graph
    bar_width = 0.35
    models_list = list(training_accuracies.keys())
    x = range(len(models_list))

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Plot the bars for training and testing accuracies
    plt.bar(x, list(training_accuracies.values()), width=bar_width, color='blue', label='Training Accuracy', align='center')
    plt.bar([i + bar_width for i in x], list(testing_accuracies.values()), width=bar_width, color='red', label='Testing Accuracy', align='center')

    # Adding labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Training vs Testing Accuracy for Each Model')

    # Set x-axis labels
    plt.xticks([i + bar_width / 2 for i in x], models_list)

    # Add a legend
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig('static/training_testing_accuracy_plot.png', dpi=300)
    plt.close()  # Close the plot after saving

# Call the function to plot and save the accuracy plot
plot_training_testing_accuracies()

# Flask Routes
@app.route('/')
def index():
    return render_template('indexx.html')

@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('best_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    if request.method == 'POST':
        data = request.get_json()  # Get JSON data
        text = data.get('text')    # Extract 'text' field from the JSON

        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)

        return jsonify({'bullying': int(prediction[0])})

@app.route('/accuracy')
def accuracy():
    return jsonify(accuracy_scores)

if __name__ == '__main__':
    app.run(debug=True)

