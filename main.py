import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px  # Product Scoresfig = px.histogram(df, x="Score")
from nltk.corpus import stopwords  # Create stopword list:
from wordcloud import WordCloud, STOPWORDS

from nltk.sentiment import SentimentIntensityAnalyzer

import nltk

nltk.download('vader_lexicon')


def is_positive(tweet: str) -> bool:
    """True if tweet has positive compound sentiment, False otherwise."""
    return sia.polarity_scores(tweet)["compound"] > 0


def filter_positive(row):
    if is_positive(row['Text']):
        return True
    else:
        return False


def filter_negative(row):
    if is_positive(row['Text']):
        return False
    else:
        return True


sia = SentimentIntensityAnalyzer()

df = pd.read_csv('Reviews.csv')
df.head()


color = sns.color_palette()

fig = px.histogram(df, x="Score")
fig.update_traces(marker_color="turquoise", marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Score')
# fig.show()


stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
textt = " ".join(review for review in df.Text)
wordcloud = WordCloud(stopwords=stopwords).generate(textt)

print(wordcloud.words_.keys())

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()

df['sentiment'] = df['Score'].apply(lambda rating: +1 if rating >= 3 else -1)


positive_bools = df.apply(filter_positive, axis=1)
positive = df[positive_bools]
negative_bools = df.apply(filter_negative, axis=1)
negative = df[negative_bools]

print(len(positive), len(negative))

stopwords = set(STOPWORDS)
# good and great removed because they were included in negative sentiment
pos = " ".join(review for review in positive.Summary)
stopwords.update(["br", "href", "good", "great"])
wordcloud2 = WordCloud(stopwords=stopwords).generate(pos)
plt.imshow(wordcloud2, interpolation='bilinear')
plt.axis("off")
plt.show()

neg = " ".join(review for review in negative.Summary)
wordcloud3 = WordCloud(stopwords=stopwords).generate(neg)
plt.imshow(wordcloud3, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud33.png')
plt.show()

df['sentimentt'] = df['sentiment'].replace({-1: 'negative'})
df['sentimentt'] = df['sentimentt'].replace({1: 'positive'})


fig = px.histogram(df, x="sentimentt")
fig.update_traces(marker_color="indianred", marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()
