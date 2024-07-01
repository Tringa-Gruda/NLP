import pandas as pd
#0.60
import re
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import TruncatedSVD

# Load the dataset
data = pd.read_excel("C:/Users/tring/OneDrive/Desktop/Homework/IBS 6/NLP/Dataset.xlsx")

# Extracting only the year 2020 data
data['Year'] = pd.to_datetime(data['date']).dt.year
data_2020 = data[data['Year'] == 2020]

# Display the filtered dataframe
print(data_2020.head())

# Save the filtered data to a new Excel file
data_2020.to_excel('C:/Users/tring/OneDrive/Desktop/Homework/IBS 6/NLP/Filtered_Data_2020.xlsx', index=False)

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

# Display the first few rows to verify the tokenization
print(data_2020[['clean_text', 'tokens']].head())

# Assuming you have a column 'category' for classification labels
label_encoder = LabelEncoder()
data_2020['label'] = label_encoder.fit_transform(data_2020['category'])

# Display the first few rows to verify the label encoding
print(data_2020[['category', 'label']].head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data_2020['clean_text'], data_2020['label'], test_size=0.2, random_state=42)

# Display the size of training and testing sets
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Vectorize the text data using TF-IDF with limited features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Reduce dimensions with TruncatedSVD
svd = TruncatedSVD(n_components=100)
X_train_reduced = svd.fit_transform(X_train_tfidf)
X_test_reduced = svd.transform(X_test_tfidf)

# Display the shape of the TF-IDF matrices
print(f"TF-IDF training matrix shape: {X_train_reduced.shape}")
print(f"TF-IDF testing matrix shape: {X_test_reduced.shape}")

# Initialize the Perceptron model
model = Perceptron()

# Train the model
model.fit(X_train_reduced, y_train)

# Display the model parameters
print(model)

# Make predictions on the test set
y_pred = model.predict(X_test_reduced)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print detailed classification report
print(classification_report(y_test, y_pred))
