# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

import inspect
import textwrap

import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import nltk

analyzer=SentimentIntensityAnalyzer()


nltk.download("all")

custom_css = """
            <style>
                .custom-column {
                    margin-right: 30px;
                    display: inline-block;
                }
                .heading-column {
                    margin-right: 30px;
                    font-weight: bold;
                    display: inline-block;
                }
                .data-column {
                    margin-right: 30px;
                    display: inline-block;
                }
            </style>
        """


def get_vader_score(text):
    text=str(text)
    # Get polarity scores for the text
    sentiment_score = analyzer.polarity_scores(text)
    return sentiment_score['compound']  # Returning the compound score

def categorize_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"n't", " not", text)  # Expand contractions
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text


st.set_page_config(page_title="sentiment",layout="wide")

st.header("Sentiment analysis")

with st.expander("Textblob Analysis"):
    data=st.text_input("Enter your review here: ")
    if data is not None:
        pre=preprocess_text(data)
        blob_data=TextBlob(pre)
        polarity=blob_data.polarity
        subjectivity=blob_data.subjectivity
        if polarity < 1 and polarity >0:
            sentiment="positive"
        elif polarity <0:
            sentiment="negative"
        else:
            sentiment="neutral"

        if subjectivity < 0.5:
            subjective="low"
        else:
            subjective="high"
            st.write("polarity: ",sentiment)
            st.write("Score: ",polarity)
            st.write("subjectivity: ",subjective)
            st.write("Score: ",subjectivity)

with st.expander("Vader analysis"):
    dataset=st.file_uploader("Upload or drag file",type="csv")
    if dataset is not None:
        df=pd.read_csv(dataset)
        reviewsdata= df['reviews.text']
        df['score'] = df['reviews.text'].apply(get_vader_score)
        df['sentiment'] = df['score'].apply(categorize_sentiment)
        text_values_list = []
        text_values_df = pd.DataFrame(columns=['Score', 'Review', 'Value of Sentiment In text'])
        for index, row in df.iterrows():
                text_values_list.append({'Score': row['score'],'Review': row['reviews.text'],'Value of Sentiment In text': row['sentiment']})
        text_values_df = pd.DataFrame(text_values_list)
        st.write(text_values_df)

with st.expander("filter"):
    filtrate=st.selectbox("filters",options=["Positive","Negative","Neutral"])
    if filtrate in ["Positive", "Negative", "Neutral"]:
        filtered_df = text_values_df[text_values_df['Value of Sentiment In text'] == filtrate]
        st.write(filtered_df) 





