# Tourist Accommodation Review Sentiment Analysis
This code performs sentiment analysis on a dataset of tourist accommodation reviews. The dataset, tourist_accommodation_reviews.csv, is provided alongside this code.

# Requirements
    pandas
    matplotlib
    seaborn
    wordcloud
    nltk

# Getting Started
    Download and install the required libraries.
    Download the stopwords, punkt, wordnet, omw-1.4, and vader_lexicon NLTK corpora by running nltk.download(["stopwords", "punkt", "wordnet", "omw-1.4", "vader_lexicon"]).
    Run the code to perform sentiment analysis on the reviews in the dataset.

# Functions
mostReviewedHotels(df)
This function takes a dataframe as input and filters it to only include the 30 hotels with the most reviews.

reviewAnalyzer(sia, reviews)
This function takes a SentimentIntensityAnalyzer object and a list of reviews as input and returns four lists containing the compound, negative, neutral, and positive sentiment scores for each review.

# Results
The code calculates some basic statistics for the sentiment score columns and creates four histograms to visualize the distribution of the sentiment scores.
