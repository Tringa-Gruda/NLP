import pandas as pd
import re
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
data = pd.read_excel("C:/Users/tring/OneDrive/Desktop/Homework/IBS 6/NLP/Dataset.xlsx")

# Extracting only the year 2020 data
data['Year'] = pd.to_datetime(data['date']).dt.year
data_2020 = data[data['Year'] == 2020]

# Define Albanian stop words
albanian_stopwords = set([
    'a', 'apo', 'asnje', 'asnjë', 'asgje', 'asgjë', 'unë', 'ti', 'ai', 'ajo',
    'ne', 'ju', 'ata', 'ato', 'ca', 'disa', 'deri', 'gjer', 'derisa', 'gjersa',
    'dhe', 'e', 'edhe', 'o', 'ose', 'i', 'jam', 'je', 'është', 'eshte', 
    'jemi', 'jeni', 'janë', 'jane', 'ishte', 'kam', 'ke', 'ka', 'kemi', 'keni',
    'kanë', 'kishte', 'kishim', 'kishin', 'kishit', 'kaq', 'këtë', 'kete',
    'me', 'më', 'mu', 'në', 'të', 'së', 'nëse', 'nese', 'nuk', 'pa', 'pas',
    'pasi', 'për', 'per', 'prej', 'që', 'qe', 'sa', 'se', 'seç', 'si', 'tij',
    'saj', 'tyre', 'imja', 'jotja', 'atij', 'asaj', 'ytja', 'juaja', 'si',
    'une', 'tek', 'ndërkohë', 'nderkohe', 'por', 'megjithatë', 'megjithate',
    'ndërsa', 'ndersa', 'përkundrazi', 'perkundrazi', 'prejse', 'përderisa',
    'perderisa', 'gjithashtu', 'poashtu', 'pra', 'mos', 'ndërmjet', 'mes', 
    'ndermjet', 'midis', 'mirëpo', 'mirepo', 'kur', 'u', 'aty', 'këtu',
    'ishim', 'ishin', 'ishit', 'isha', 'ishe', 'ishte'])

# Function to clean text
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = ' '.join([word for word in text.split() if word not in albanian_stopwords])
        return text
    else:
        return ''

# Apply the clean_text function to your article content column
data_2020['clean_text'] = data_2020['content'].apply(clean_text)

# Tokenize the text
data_2020['tokens'] = data_2020['clean_text'].apply(word_tokenize)

# Assuming you have a column 'category' for classification labels
label_encoder = LabelEncoder()
label_encoder.fit(data_2020['category'])
data_2020['label'] = label_encoder.transform(data_2020['category'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data_2020['clean_text'], data_2020['label'], test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF with limited features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Reduce dimensions with TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_test_reduced = svd.transform(X_test_tfidf)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_reduced, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_reduced)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Save the model, vectorizer, SVD transformer, and label encoder
joblib.dump(model, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(svd, 'svd_transformer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Function to predict category for input text
def predict_category(article_text):
    clean_article = clean_text(article_text)
    tokens = word_tokenize(clean_article)
    article_tfidf = vectorizer.transform([' '.join(tokens)])
    article_reduced = svd.transform(article_tfidf)
    prediction = model.predict(article_reduced)
    category = label_encoder.inverse_transform(prediction)
    return category[0]

# Load the saved model, vectorizer, SVD transformer, and label encoder
model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
svd = joblib.load('svd_transformer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Take user input from the console
while True:
    article_text = input("Enter the content of the news article (or 'exit' to quit): ")
    if article_text.lower() == 'exit':
        break
    predicted_category = predict_category(article_text)
    print(f"The predicted category is: {predicted_category}")
