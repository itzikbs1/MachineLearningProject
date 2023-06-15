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


# Examine the correlation between personality types codes
# Split type columns into four binary columns
split_df = mbti_df[['type', 'posts']].copy()


split_df['E-I'] = split_df['type'].str.extract('(.)[N,S]', 1)
split_df['N-S'] = split_df['type'].str.extract('[E,I](.)[F,T]', 1)
split_df['T-F'] = split_df['type'].str.extract('[N,S](.)[J,P]', 1)
split_df['J-P'] = split_df['type'].str.extract('[F,T](.)', 1)
# split_df.head()
# print(split_df.head())

# Encode letters to numeric values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

encoded_df = split_df[['type', 'posts']].copy()
encoded_df['E0-I1'] = le.fit_transform(split_df['E-I'])
encoded_df['N0-S1'] = le.fit_transform(split_df['N-S'])
encoded_df['F0-T1'] = le.fit_transform(split_df['T-F'])
encoded_df['J0-P1'] = le.fit_transform(split_df['J-P'])

# encoded_df.head()
# print(encoded_df.head())

# Define X and y
X = encoded_df["posts"].values
y_all = encoded_df.drop(columns=['type', 'posts'])

# Split training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_all_train, y_all_test = train_test_split(X, y_all, random_state=42)

# Define TFIDF verctorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=17000,
    min_df=7,
    max_df=0.8,
    ngram_range=(1,3),
)

# create vectors for X
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# create log reg model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs', random_state=1)

# implement random oversampling
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=1)

y_EI_train = y_all_train['E0-I1']
y_EI_test = y_all_test['E0-I1']

X_resampled_ros, y_EI_resampled_ros = ros.fit_resample(X_train, y_EI_train)
# Fit E-I combination with oversampled x_train and y_EI_train
classifier.fit(X_resampled_ros, y_EI_resampled_ros)
# Predict outcomes for test data set
y_EI_pred_ros = classifier.predict(X_test)
EI_result = pd.DataFrame({"Prediction": y_EI_pred_ros, "Actual": y_EI_test})


# resample N-S combination
y_NS_train = y_all_train['N0-S1']
y_NS_test = y_all_test['N0-S1']

X_resampled_ros, y_NS_resampled_ros = ros.fit_resample(X_train, y_NS_train)

classifier.fit(X_resampled_ros, y_NS_resampled_ros)

# Predict outcomes for test data set
y_NS_pred_ros = classifier.predict(X_test)
NS_result = pd.DataFrame({"Prediction": y_NS_pred_ros, "Actual": y_NS_test})
# NS_result.head(5)

# resample F-T combination
y_FT_train = y_all_train['F0-T1']
y_FT_test = y_all_test['F0-T1']

X_resampled_ros, y_FT_resampled_ros = ros.fit_resample(X_train, y_FT_train)
# Fit F-T combination with oversampled x_train and y_FT_train
classifier.fit(X_resampled_ros, y_FT_resampled_ros)
# Predict outcomes for test data set
y_FT_pred_ros = classifier.predict(X_test)
FT_result = pd.DataFrame({"Prediction": y_FT_pred_ros, "Actual": y_FT_test})
# FT_result.head(5)
# resample J-P combination
y_JP_train = y_all_train['J0-P1']
y_JP_test = y_all_test['J0-P1']

X_resampled_ros, y_JP_resampled_ros = ros.fit_resample(X_train, y_JP_train)
# Fit J-P combination with oversampled x_train and y_JP_train
classifier.fit(X_resampled_ros, y_JP_resampled_ros)

# Predict outcomes for test data set
y_JP_pred_ros = classifier.predict(X_test)
JP_result = pd.DataFrame({"Prediction": y_JP_pred_ros, "Actual": y_JP_test})
# JP_result.head(5)

# Calculate accuracy score for each group
from sklearn.metrics import accuracy_score
print(f" Logistic regression model accuracy for E-I: {accuracy_score(y_EI_test, y_EI_pred_ros):.3f}")
print(f" Logistic regression model accuracy for N-S: {accuracy_score(y_NS_test, y_NS_pred_ros):.3f}")
print(f" Logistic regression model accuracy for F-T: {accuracy_score(y_FT_test, y_FT_pred_ros):.3f}")
print(f" Logistic regression model accuracy for J-P: {accuracy_score(y_JP_test, y_JP_pred_ros):.3f}")