import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from DataSet import DataSet
from LogisticRegressionModel import LogisticRegressionModel
from RandomForestModel import RandomForestModel
from NaiveBaseModel import  NaiveBaseModel
# Define TFIDF verctorizer
# vectorizer = TfidfVectorizer(
#     max_features=17000,
#     min_df=7,
#     max_df=0.8,
#     stop_words="english",
#     ngram_range=(1, 3),
# )
if __name__ == '__main__':

    dataset = DataSet()
    # dataset_vectors = dataset.handle_data() # X_train, X_test, y_all_train, y_all_test
    X_train, X_test, y_all_train, y_all_test, vectorizer = dataset.handle_data()
    lrm = LogisticRegressionModel()
    input_user = input("Please Enter text: ")

     # clean the text with regex
    replacements = [
        (r"(http.*?\s)", " "),
        (r"[^\w\s]", " "),
        (r"\_", " "),
        (r"\d+", " ")]

    for old, new in replacements:
        input_user = re.sub(old,new, input_user)

    # input_user = [input_user]
    # print("input_user ", input_user)
    # vectorize the cleaned text
    input_user_Vectorized = vectorizer.transform([input_user])
    ei_lrm = lrm.EIModel(X_train, X_test, y_all_train, y_all_test)
    ei_prediction = ei_lrm[0].predict(input_user_Vectorized)

    ns_lrm = lrm.NSModel(X_train, X_test, y_all_train, y_all_test)
    ns_prediction = ns_lrm[0].predict(input_user_Vectorized)

    ft_lrm = lrm.FTModel(X_train, X_test, y_all_train, y_all_test)
    ft_prediction = ft_lrm[0].predict(input_user_Vectorized)

    jp_lrm = lrm.JPModel(X_train, X_test, y_all_train, y_all_test)
    jp_prediction = jp_lrm[0].predict(input_user_Vectorized)


    # convert the prediction result from 1 and 0 to letters
    output_EI = 'E' if ei_prediction == 0 else "I"
    output_NS = 'N' if ns_prediction == 0 else "S"
    output_FT = 'F' if ft_prediction == 0 else "T"
    output_JP = 'J' if jp_prediction == 0 else "P"
    prediction_text = (f'{output_EI}{output_NS}{output_FT}{output_JP}')
    print(prediction_text)