import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize  # Import the word_tokenize function
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# Step 1: Read the Excel data
data = pd.read_excel("CleanedTrainingData.xlsx")

# Step 2: Preprocess the text
# Convert tweet content to string and tokenize
data['tweetcontent'] = data['tweetcontent'].astype(str).apply(word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['tweetcontent'] = data['tweetcontent'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatize the words
lemmatizer = WordNetLemmatizer()
data['tweetcontent'] = data['tweetcontent'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Convert list of tokens back to string
data['tweetcontent'] = data['tweetcontent'].apply(' '.join)

print(data.head(1))

#Next Steps: 
#Feed the pre-processed data to the model
#Train an existing model on our pre-processed data
#Use that model to get ouputs as needed