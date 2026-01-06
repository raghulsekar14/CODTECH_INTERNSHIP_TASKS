import pandas as pd
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download English stopwords
nltk.download('stopwords')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, "sentiment.csv")

# Load the dataset
data = pd.read_csv(csv_path)

# Initialize stemmer for word stemming
stemmer = PorterStemmer()

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Function to preprocess text data
def preprocess_text(text):
    text = str(text).lower()                    
    text = re.sub('[^a-z]', ' ', text)          
    words = text.split()                        
    words = [stemmer.stem(word)                
             for word in words 
             if word not in stop_words]        
    return ' '.join(words)                      

# Apply preprocessing to tweets column
data['clean_tweets'] = data['tweets'].apply(preprocess_text)

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer(max_features=3000)

# Convert text data into numerical features
X = tfidf.fit_transform(data['clean_tweets'])

# Target labels (sentiment classes)
y = data['labels']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Initialize Logistic Regression model for multiclass classification
model = LogisticRegression(multi_class='multinomial', max_iter=1000)

# Train the model on training data
model.fit(X_train, y_train)

# Predict sentiments for test data
y_pred = model.predict(X_test)

# Print model accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
print()

# Print detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
print()

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Sample tweets for prediction
sample_tweets = [
    "ChatGPT is awesome",
    "This is terrible",
    "OpenAI research update"
]

# Preprocess sample tweets
processed = [preprocess_text(t) for t in sample_tweets]

# Convert sample tweets into TF-IDF vectors
vectors = tfidf.transform(processed)

# Predict sentiment for sample tweets
predictions = model.predict(vectors)

print()

# Display tweet with predicted sentiment
for tweet, sentiment in zip(sample_tweets, predictions):
    print(tweet, "->", sentiment)
