import re

import nltk
import pandas as pd

from nltk.corpus import stopwords
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')

mbti_df = pd.read_csv('mbti_1.csv')


mbti_df.head()
# print(df)
# print(df.posts[0])
# print(df.head())  # Display the first few rows
# print(df.info())  # Get information about the DataFrame
mbti_df["posts"] = mbti_df["posts"].str.lower()       #converts text in posts to lowercase as it is preferred in nlp
for i in range(len(mbti_df)):
    post_temp = mbti_df._get_value(i, 'posts')
    pattern = re.compile(
        r'https?://[a-zA-Z0-9./-]*/[a-zA-Z0-9?=_.]*[_0-9.a-zA-Z/-]*')  # to match url links present in the post
    # print("post_temp ", post_temp)
    post_temp = re.sub(pattern, ' ', post_temp)  # to replace that url link with space
    # print("post_temp ", post_temp)
    # break
    mbti_df._set_value(i, 'posts', post_temp)

# print(mbti_df.posts[2])
# print("************************************************************************************************")
for i in range(len(mbti_df)):
    post_temp = mbti_df._get_value(i, 'posts')
    pattern = re.compile(r'[0-9]')  # to match numbers from 0 to 9
    post_temp = re.sub(pattern, ' ', post_temp)  # to replace them with space
    pattern = re.compile('\W+')  # to match alphanumeric characters
    post_temp = re.sub(pattern, ' ', post_temp)  # to replace them with space
    pattern = re.compile(r'[_+]')
    post_temp = re.sub(pattern, ' ', post_temp)
    mbti_df._set_value(i, 'posts', post_temp)

# print(mbti_df.posts[2])
for i in range(len(mbti_df)):
    post_temp=mbti_df._get_value(i, 'posts')
    pattern = re.compile('\s+')                                     #to match multiple whitespaces
    post_temp= re.sub(pattern, ' ', post_temp)                      #to replace them with single whitespace
    mbti_df._set_value(i, 'posts', post_temp)

# print(mbti_df.posts[2])

remove_words = stopwords.words("english")
for i in range(mbti_df.shape[0]):
    post_temp=mbti_df._get_value(i, 'posts')
    post_temp=" ".join([w for w in post_temp.split(' ') if w not in remove_words])    #to remove stopwords
    mbti_df._set_value(i, 'posts', post_temp)

# print(mbti_df.posts[2])

for i in range(mbti_df.shape[0]):
    post_temp=mbti_df._get_value(i, 'posts')
    post_temp=" ".join([lemmatizer.lemmatize(w) for w in post_temp.split(' ')])   #to implement lemmetization i.e. to group together different forms of a word
    mbti_df._set_value(i, 'posts', post_temp)

