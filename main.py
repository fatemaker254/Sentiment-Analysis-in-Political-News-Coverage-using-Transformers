# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from transformers import pipeline
import torch
import re
import warnings
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
val = pd.read_csv('val.csv')

# Display the first few rows of each dataset
print('Train Dataset:\n', train.head())
print('\nTest Dataset:\n', test.head())
print('\nValidation Dataset:\n', val.head())

# Summary statistics of train dataset
print('Train Dataset Info:')
train.info()

# Summary statistics of test dataset
print('\nTest Dataset Info:')
test.info()

# Summary statistics of validation dataset
print('\nValidation Dataset Info:')
val.info()

# Check for duplicate entries
print('\nDuplicate entries in Train:', train.duplicated().sum())
print('Duplicate entries in Test:', test.duplicated().sum())
print('Duplicate entries in Validation:', val.duplicated().sum())

# Describe numerical features
print('\nNumerical Features Summary:')
print(train.describe())
# Sentiment distribution in train data
plt.figure(figsize=(8,6))
sns.countplot(x='sentiment', data=train, palette='coolwarm')
plt.title('Sentiment Distribution in Training Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
# Sentiment distribution across parties
plt.figure(figsize=(14,8))
sns.countplot(x='party', hue='sentiment', data=train, palette='coolwarm')
plt.title('Sentiment Distribution by Political Party')
plt.xlabel('Political Party')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Sentiment')
plt.show()
# Add a new column for tweet length
train['tweet_length'] = train['tweet_text'].apply(lambda x: len(x.split()))
test['tweet_length'] = test['tweet_text'].apply(lambda x: len(x.split()))
val['tweet_length'] = val['tweet_text'].apply(lambda x: len(x.split()))

# Plot distribution of tweet lengths
plt.figure(figsize=(10,6))
sns.histplot(train['tweet_length'], bins=30, kde=True, color='teal')
plt.title('Distribution of Tweet Lengths in Training Data')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

# Boxplot of tweet length by sentiment
plt.figure(figsize=(8,6))
sns.boxplot(x='sentiment', y='tweet_length', data=train, palette='coolwarm')
plt.title('Tweet Length by Sentiment')
plt.xlabel('Sentiment')
plt.ylabel('Tweet Length (Words)')
plt.show()

# Convert timestamp to datetime
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])
val['timestamp'] = pd.to_datetime(val['timestamp'])

# Set timestamp as index for train data
train.set_index('timestamp', inplace=True)

# Resample to weekly frequency and count sentiments
weekly_sentiment = train.groupby([pd.Grouper(freq='W'), 'sentiment']).size().unstack().fillna(0)


# Plot time series of sentiments
plt.figure(figsize=(16, 8))
weekly_sentiment.plot()
plt.title('Weekly Sentiment Counts Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.show()

# Function to generate word cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(' '.join(text))
    plt.figure(figsize=(15,7.5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=20)
    plt.axis('off')
    plt.show()

# Generate word clouds for each sentiment
for sentiment in ['positive', 'neutral', 'negative']:
    subset = train[train['sentiment'] == sentiment]
    generate_wordcloud(subset['tweet_text'], f'Word Cloud for {sentiment.capitalize()} Sentiment')

# Combine all datasets for preprocessing
data = pd.concat([train.reset_index(), test.reset_index(), val.reset_index()], ignore_index=True)

# Function to clean tweet text
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    return text

# Apply cleaning
data['clean_text'] = data['tweet_text'].apply(clean_text)

# Display cleaned text
data[['tweet_text', 'clean_text']].head()

# Check for missing values
print('Missing Values in Combined Data:\n', data.isnull().sum())

# Since there are no missing values in the provided data, we proceed
# If there were, we could handle them as follows:
# data = data.dropna()
# or
# data['column_name'].fillna('value', inplace=True)

# Verify no missing values
print('\nAfter Handling Missing Values:\n', data.isnull().sum())

# Encode 'party' using Label Encoding
from sklearn.preprocessing import LabelEncoder

le_party = LabelEncoder()
data['party_encoded'] = le_party.fit_transform(data['party'])

# Map sentiment to numerical values for evaluation
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
data['sentiment_encoded'] = data['sentiment'].map(sentiment_mapping)

# Split back into train, test, val
train = data.iloc[:500].copy()
test = data.iloc[500:550].copy()
val = data.iloc[550:600].copy()

# Display encoded features
train[['party', 'party_encoded', 'sentiment', 'sentiment_encoded']].head()

# Initialize sentiment analysis pipeline with a pre-trained model
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Test the pipeline on a sample tweet
sample_tweet = train['tweet_text'].iloc[0]
print('Sample Tweet:', sample_tweet)
print('Sentiment Analysis Result:', sentiment_pipeline(sample_tweet))

# Function to perform sentiment analysis using the pipeline
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]  # Truncate to 512 tokens
        return result['label'], result['score']
    except Exception as e:
        return 'ERROR', 0

# Apply sentiment analysis to training set
train[['predicted_sentiment', 'confidence']] = train['tweet_text'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# Apply sentiment analysis to validation set
val[['predicted_sentiment', 'confidence']] = val['tweet_text'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# Apply sentiment analysis to test set
test[['predicted_sentiment', 'confidence']] = test['tweet_text'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# Display results
train[['tweet_text', 'predicted_sentiment', 'confidence']].head()

val['predicted_sentiment'] = val['tweet_text'].apply(lambda x: analyze_sentiment(x)[0])

# Convert predicted sentiment to lowercase
val['predicted_sentiment'] = val['predicted_sentiment'].str.lower()

# Convert to categorical AFTER converting to lowercase AND adding the category
val['predicted_sentiment'] = pd.Categorical(val['predicted_sentiment'], categories=['negative', 'neutral', 'positive'])




# Check for missing predicted sentiments and handle them (using the new approach)
if val['predicted_sentiment'].isnull().any():
    print("Warning: Missing predicted sentiments found. Filling with 'NEUTRAL'.")
    val['predicted_sentiment'] = val['predicted_sentiment'].fillna('neutral')


# Print unique values to verify
print("Unique values in val['sentiment']:", val['sentiment'].unique())
print("Unique values in val['predicted_sentiment']:", val['predicted_sentiment'].unique())


# Generate classification report
print(classification_report(val['sentiment'], val['predicted_sentiment'], target_names=['negative', 'neutral', 'positive'])) #Explicitly stating categories here

# Generate confusion matrix using lowercase labels
cm = confusion_matrix(val['sentiment'], val['predicted_sentiment'], labels=['negative', 'neutral', 'positive'])  # Use lowercase labels

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])  # Lowercase labels for ticks
plt.title('Confusion Matrix for Validation Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot confidence distribution by sentiment
plt.figure(figsize=(10,6))
sns.boxplot(x='predicted_sentiment', y='confidence', data=train, palette='coolwarm')
plt.title('Confidence Scores by Predicted Sentiment')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Confidence Score')
plt.show()

# Average confidence score by party and sentiment
plt.figure(figsize=(14,8))
sns.barplot(x='party', y='confidence', hue='predicted_sentiment', data=train, palette='coolwarm')
plt.title('Average Confidence Score by Party and Sentiment')
plt.xlabel('Political Party')
plt.ylabel('Average Confidence Score')
plt.xticks(rotation=45)
plt.legend(title='Predicted Sentiment')
plt.show()
accuracy = accuracy_score(val['sentiment'], val['predicted_sentiment'])
print("Validation Accuracy:", round(accuracy * 100, 2), "%")