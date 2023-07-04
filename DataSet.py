import re

import nltk
import pandas as pd

from nltk.corpus import stopwords

nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


class DataSet:

    def __init__(self):
        self.mbti_df = pd.read_csv('mbti_1.csv')

    def handle_data(self):
        self.mbti_df["posts"] = self.mbti_df["posts"].str.lower()
        for i in range(len(self.mbti_df)):
            post_temp = self.mbti_df._get_value(i, 'posts')
            pattern = re.compile(
                r'https?://[a-zA-Z0-9./-]*/[a-zA-Z0-9?=_.]*[_0-9.a-zA-Z/-]*')  # to match url links present in the post
            post_temp = re.sub(pattern, ' ', post_temp)  # to replace that url link with space
            self.mbti_df._set_value(i, 'posts', post_temp)

        for i in range(len(self.mbti_df)):
            post_temp = self.mbti_df._get_value(i, 'posts')
            pattern = re.compile(r'[0-9]')  # to match numbers from 0 to 9
            post_temp = re.sub(pattern, ' ', post_temp)  # to replace them with space
            pattern = re.compile('\W+')  # to match non alphanumeric characters
            post_temp = re.sub(pattern, ' ', post_temp)  # to replace them with space
            pattern = re.compile(r'[_+]')
            post_temp = re.sub(pattern, ' ', post_temp)
            self.mbti_df._set_value(i, 'posts', post_temp)

        for i in range(len(self.mbti_df)):
            post_temp = self.mbti_df._get_value(i, 'posts')
            pattern = re.compile('\s+')  # to match multiple whitespaces
            post_temp = re.sub(pattern, ' ', post_temp)  # to replace them with single whitespace
            self.mbti_df._set_value(i, 'posts', post_temp)

        remove_words = stopwords.words("english")
        for i in range(self.mbti_df.shape[0]):
            post_temp = self.mbti_df._get_value(i, 'posts')
            post_temp = " ".join([w for w in post_temp.split(' ') if w not in remove_words])  # to remove stopwords
            self.mbti_df._set_value(i, 'posts', post_temp)

        for i in range(self.mbti_df.shape[0]):
            post_temp = self.mbti_df._get_value(i, 'posts')
            post_temp = " ".join([lemmatizer.lemmatize(w) for w in post_temp.split(' ')])  # to implement lemmetization. group together different forms of a word
            self.mbti_df._set_value(i, 'posts', post_temp)

        copy = self.mbti_df[['type', 'posts']].copy()

        copy['E-I'] = copy['type'].str.extract('(.)[N,S]', 1)
        copy['N-S'] = copy['type'].str.extract('[E,I](.)[F,T]', 1)
        copy['T-F'] = copy['type'].str.extract('[N,S](.)[J,P]', 1)
        copy['J-P'] = copy['type'].str.extract('[F,T](.)', 1)

        # Encode letters to numeric values
        le = LabelEncoder()
        labeled_data = copy[['type', 'posts']].copy()
        labeled_data['E0-I1'] = le.fit_transform(copy['E-I'])
        labeled_data['N0-S1'] = le.fit_transform(copy['N-S'])
        labeled_data['F0-T1'] = le.fit_transform(copy['T-F'])
        labeled_data['J0-P1'] = le.fit_transform(copy['J-P'])

        # Define X and y
        X = labeled_data["posts"].values
        Y = labeled_data.drop(columns=['type', 'posts'])

        # Split training and testing dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=42) # X_train and X_test will be all the posts divided to
        # train and test. y_all_train and y_all_test will be the target labels i.e the MBTI type letters divided to the train and test. 25% goes to test. 75% to train.

        # Define TFIDF verctorizer
        vectorizer = TfidfVectorizer(
            max_features=17000,
            min_df=7,
            max_df=0.8,
            ngram_range=(1, 3),
        )

        # create vectors for X. We need to vectorize all the text for the ML models.
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        return X_train, X_test, y_train, y_test, vectorizer
