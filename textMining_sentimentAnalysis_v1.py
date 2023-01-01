# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:51:39 2022

@author: imyaash-admin
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud as wc
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist

# Set font scale for seaborn plots
sns.set(font_scale = 0.7)

# Download NLTK corpora
nltk.download(["stopwords", "punkt", "wordnet", "omw-1.4", "vader_lexicon"])

# Load the dataset
df = pd.read_csv("Datasets/tourist_accomodation_reviews.csv")

def mostReviewedHotels(df):
    # Create empty lists for the hotel names and number of reviews
    hotel = []
    reviews = []
    # Iterate over the unique hotel names in the data
    for i in df["Hotel/Restaurant name"].unique():
        # Add the current hotel name to the list
        hotel.append(i)

        # Count the number of reviews for the current hotel and add it to the list
        reviews.append(df[df["Hotel/Restaurant name"] == i]["Review"].count())

    # Create a dataframe with the hotel names and review counts
    hotelReviews = pd.concat([pd.Series(hotel, name = "Hotel"), pd.Series(reviews, name = "No. of Reviews")], axis = 1)

    # Sort the dataframe by the number of reviews in descending order and keep only the top 30 hotels
    hotelReviews = hotelReviews.sort_values(by = "No. of Reviews", ascending = False).head(30)

    # Create an empty dataframe for the filtered reviews
    dfNew = pd.DataFrame()

    # Iterate over the hotels in the filtered dataframe
    for i in hotelReviews.Hotel:
        # Add the reviews for the current hotel to the filtered dataframe
        dfNew = pd.concat([dfNew, df[df["Hotel/Restaurant name"] == i]], axis = 0)

    # Return the filtered dataframe
    return dfNew

# Filter the data to only include only the 30 most reviewed hotels
df = mostReviewedHotels(df)


def reviewAnalyzer(sia, reviews):
    # Create empty lists for the sentiment scores
    compound = []
    negetive = []
    neutral = []
    positive = []
    # Iterate over the reviews
    for i in reviews:
        # Compute the sentiment scores for the current review and add them to the corresponding lists
        compound.append(sia.polarity_scores(i)["compound"])
        negetive.append(sia.polarity_scores(i)["neg"])
        neutral.append(sia.polarity_scores(i)["neu"])
        positive.append(sia.polarity_scores(i)["pos"])
    return compound, negetive, neutral, positive

# Create a instance of the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# Compute the sentiment scores for each review in the dataset
df["Compound"], df["Negetive"], df["Neutral"], df["Positive"] = reviewAnalyzer(sia, df["Review"])

# Create a list of the sentiment score columns
res = ["Compound", "Negetive", "Neutral", "Positive"]

# Calculate some basic statistics for the sentiment score columns
df[res].describe()

# create a figure with four subplots arranged in a 2x2 grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
# set the figure title
fig.suptitle("Distribution of Sentiments")
# create a histogram of the data in the first grid of the frame and set the title of the subplot
ax1.hist(df[res[0]])
ax1.set_title(res[0])
# create a histogram of the data in the second grid of the frame and set the title of the subplot
ax2.hist(df[res[1]])
ax2.set_title(res[1])
# create a histogram of the data in the third grid of the frame and set the title of the subplot
ax3.hist(df[res[2]])
ax3.set_title(res[2])
# create a histogram of the data in the fourth grid of the frame and set the title of the subplot
ax4.hist(df[res[3]])
ax4.set_title(res[3])
# ensure that x-axis and y-axis labels are only displayed on the bottom and leftmost subplots, respectively
for ax in fig.get_axes():
    ax.label_outer()


# Count the number of negetive reviews for each hotel in the dataset
(df["Compound"] <= 0).groupby(df["Hotel/Restaurant name"]).sum()
# Calculate the percentage of negetive reviews for each hotel in the dataset
negPercent = pd.DataFrame((df["Compound"] <= 0).groupby(df["Hotel/Restaurant name"]).sum() / df["Hotel/Restaurant name"].groupby(df["Hotel/Restaurant name"]).count() * 100, columns = ["% Negetive Reviews"]).sort_values(by = "% Negetive Reviews", ascending = False)
negPercent
# Create a bar plot of the percentage of negetive reviews
sns.barplot(data = negPercent, x = "% Negetive Reviews", y = negPercent.index, color = "Red")

# Count the number of positive reviews for each hotel in the dataset
(df["Compound"] > 0).groupby(df["Hotel/Restaurant name"]).sum()
# Calculate the percentage of positive reviews for each hotel in the dataset
posPercent = pd.DataFrame((df["Compound"] > 0).groupby(df["Hotel/Restaurant name"]).sum() / df["Hotel/Restaurant name"].groupby(df["Hotel/Restaurant name"]).count() * 100, columns = ["% Positive Reviews"]).sort_values(by = "% Positive Reviews", ascending = False)
posPercent
# Create a bar plot of the percentage of positive reviews
sns.barplot(data = posPercent, x = "% Positive Reviews", y = posPercent.index, color = "Orange")

# Import the NLTK stop words list
stopWord = nltk.corpus.stopwords.words("english")
def preprocessor(text):
    # Tokenize the text using a regular expression that matches alphanumeric characters and single quotes
    tokenized = nltk.tokenize.RegexpTokenizer("[a-zA-Z0-9\']+").tokenize(text)
    # Create an empty list for the cleaned tokens
    cleanedToken = []
    # Loop over the tokens
    for word in tokenized:
        # If the current token is not in the NLTK stop words list, add it to the list of cleaned tokens
        if word.lower() not in stopWord:
            cleanedToken.append(word.lower())
    # Create an empty list for the stemmed tokens
    stemmedText = []
    # Loop over the cleaned tokens
    for word in cleanedToken:
        # Stem the current token using the Porter stemmer and add it to the list of stemmed tokens
        stemmedText.append(nltk.stem.PorterStemmer().stem(word))
    # Return the list of stemmed tokens
    return stemmedText

df["Processed Review"] = df["Review"].apply(preprocessor) # apply the preprocessor function to the "Review" column and save the result in a new "Processed Review" column

# Text mining for most positively reviwed hotel
posRevSub = df.loc[(df["Hotel/Restaurant name"] == posPercent.head(1).index[0]) & (df["Compound"] > 0), :] # create a new DataFrame containing only positive reviews of the "Natural Restaurant"
negRevSub = df.loc[(df["Hotel/Restaurant name"] == posPercent.head(1).index[0]) & (df["Compound"] <= 0), :] # create a new DataFrame containing only negative reviews of the "Natural Restaurant"

posToken = [word for review in posRevSub["Processed Review"] for word in review] # create a list of tokens from the positive reviews
negToken = [word for review in negRevSub["Processed Review"] for word in review] # create a list of tokens from the negative reviews

wcPos = wc(background_color = "white").generate_from_text(" ".join(posToken)) # create a word cloud from the positive tokens
plt.figure(figsize = (12, 12))
plt.imshow(wcPos, interpolation = "bilinear")
plt.axis = ("off")
plt.title("Most used positive words for " + posPercent.head(1).index[0])
plt.show()

wcNeg = wc(background_color = "white").generate_from_text(" ".join(negToken)) # create a word cloud from the negative tokens
plt.figure(figsize = (12, 12))
plt.imshow(wcNeg, interpolation = "bilinear")
plt.axis = ("off")
plt.title("Most used negetive words for " + posPercent.head(1).index[0])
plt.show()

posFreqDist = FreqDist(posToken) # create a frequency distribution of the positive tokens
posFreqDist.tabulate(10) # display the top 10 words in the distribution
negFreqDist = FreqDist(negToken) # create a frequency distribution of the negative tokens
negFreqDist.tabulate(10) # display the top 10 words in the distribution

posFreqDist.plot(10, title = "Most used positive words for " + posPercent.head(1).index[0]) # plot the top 10 words in the positive frequency distribution
negFreqDist.plot(10, title = "Most used negetive words for " + posPercent.head(1).index[0]) # plot the top 10 words in the negative frequency distribution

# Text mining for most negetively reviwed hotel
posRevSub = df.loc[(df["Hotel/Restaurant name"] == negPercent.head(1).index[0]) & (df["Compound"] > 0), :] # create a new DataFrame containing only positive reviews of the "Natural Restaurant"
negRevSub = df.loc[(df["Hotel/Restaurant name"] == negPercent.head(1).index[0]) & (df["Compound"] <= 0), :] # create a new DataFrame containing only negative reviews of the "Natural Restaurant"

posToken = [word for review in posRevSub["Processed Review"] for word in review] # create a list of tokens from the positive reviews
negToken = [word for review in negRevSub["Processed Review"] for word in review] # create a list of tokens from the negative reviews

wcPos = wc(background_color = "white").generate_from_text(" ".join(posToken)) # create a word cloud from the positive tokens
plt.figure(figsize = (12, 12))
plt.imshow(wcPos, interpolation = "bilinear")
plt.axis = ("off")
plt.title("Most used positive words for " + negPercent.head(1).index[0])
plt.show()

wcNeg = wc(background_color = "white").generate_from_text(" ".join(negToken)) # create a word cloud from the negative tokens
plt.figure(figsize = (12, 12))
plt.imshow(wcNeg, interpolation = "bilinear")
plt.axis = ("off")
plt.title("Most used negetive words for " + negPercent.head(1).index[0])
plt.show()

posFreqDist = FreqDist(posToken) # create a frequency distribution of the positive tokens
posFreqDist.tabulate(10) # display the top 10 words in the distribution
negFreqDist = FreqDist(negToken) # create a frequency distribution of the negative tokens
negFreqDist.tabulate(10) # display the top 10 words in the distribution

posFreqDist.plot(10, title = "Most used positive words for " + negPercent.head(1).index[0]) # plot the top 10 words in the positive frequency distribution
negFreqDist.plot(10, title = "Most used negetive words for " + negPercent.head(1).index[0]) # plot the top 10 words in the negative frequency distribution
