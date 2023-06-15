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
        self.mbti_df["posts"] = self.mbti_df["posts"].str.lower()       #converts text in posts to lowercase as it is preferred in nlp
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
            pattern = re.compile('\W+')  # to match alphanumeric characters
            post_temp = re.sub(pattern, ' ', post_temp)  # to replace them with space
            pattern = re.compile(r'[_+]')
            post_temp = re.sub(pattern, ' ', post_temp)
            self.mbti_df._set_value(i, 'posts', post_temp)

        for i in range(len(self.mbti_df)):
            post_temp=self.mbti_df._get_value(i, 'posts')
            pattern = re.compile('\s+')                                     #to match multiple whitespaces
            post_temp= re.sub(pattern, ' ', post_temp)                      #to replace them with single whitespace
            self.mbti_df._set_value(i, 'posts', post_temp)

        remove_words = stopwords.words("english")
        for i in range(self.mbti_df.shape[0]):
            post_temp=self.mbti_df._get_value(i, 'posts')
            post_temp=" ".join([w for w in post_temp.split(' ') if w not in remove_words])    #to remove stopwords
            self.mbti_df._set_value(i, 'posts', post_temp)

        for i in range(self.mbti_df.shape[0]):
            post_temp=self.mbti_df._get_value(i, 'posts')
            post_temp=" ".join([lemmatizer.lemmatize(w) for w in post_temp.split(' ')])   #to implement lemmetization i.e. to group together different forms of a word
            self.mbti_df._set_value(i, 'posts', post_temp)


        # Examine the correlation between personality types codes
        # Split type columns into four binary columns
        split_df = self.mbti_df[['type', 'posts']].copy()


        split_df['E-I'] = split_df['type'].str.extract('(.)[N,S]', 1)
        split_df['N-S'] = split_df['type'].str.extract('[E,I](.)[F,T]', 1)
        split_df['T-F'] = split_df['type'].str.extract('[N,S](.)[J,P]', 1)
        split_df['J-P'] = split_df['type'].str.extract('[F,T](.)', 1)

        # Encode letters to numeric values
        le = LabelEncoder()
        encoded_df = split_df[['type', 'posts']].copy()
        encoded_df['E0-I1'] = le.fit_transform(split_df['E-I'])
        encoded_df['N0-S1'] = le.fit_transform(split_df['N-S'])
        encoded_df['F0-T1'] = le.fit_transform(split_df['T-F'])
        encoded_df['J0-P1'] = le.fit_transform(split_df['J-P'])

        # Define X and y
        X = encoded_df["posts"].values
        y_all = encoded_df.drop(columns=['type', 'posts'])

        # Split training and testing dataset
        X_train, X_test, y_all_train, y_all_test = train_test_split(X, y_all, random_state=42)

        # Define TFIDF verctorizer
        vectorizer = TfidfVectorizer(
            max_features=17000,
            min_df=7,
            max_df=0.8,
            ngram_range=(1,3),
        )

        # create vectors for X
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
        return X_train, X_test, y_all_train, y_all_test, vectorizer