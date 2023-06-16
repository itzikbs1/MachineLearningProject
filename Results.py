from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from DataSet import DataSet
from LogisticRegressionModel import LogisticRegressionModel
from RandomForestModel import RandomForestModel
from NaiveBaseModel import  NaiveBaseModel
if __name__ == '__main__':

    dataset = DataSet()
    X_train, X_test, y_all_train, y_all_test, vectorizer = dataset.handle_data()

    lrm = LogisticRegressionModel()
    rfm = RandomForestModel()

    ei_lrm = lrm.EIModel(X_train, X_test, y_all_train, y_all_test)
    ns_lrm = lrm.NSModel(X_train, X_test, y_all_train, y_all_test)
    ft_lrm = lrm.FTModel(X_train, X_test, y_all_train, y_all_test)
    jp_lrm = lrm.JPModel(X_train, X_test, y_all_train, y_all_test)
    print(f" Logistic regression model accuracy for E-I: {accuracy_score(ei_lrm[1], ei_lrm[2]):.3f}")
    print(f" Logistic regression model accuracy for N-S: {accuracy_score(ns_lrm[1], ns_lrm[2]):.3f}")
    print(f" Logistic regression model accuracy for F-T: {accuracy_score(ft_lrm[1], ft_lrm[2]):.3f}")
    print(f" Logistic regression model accuracy for J-P: {accuracy_score(jp_lrm[1], jp_lrm[2]):.3f}")

    ei_rfm = rfm.EIModel(X_train, X_test, y_all_train, y_all_test)
    ns_rfm = rfm.NSModel(X_train, X_test, y_all_train, y_all_test)
    ft_rfm = rfm.FTModel(X_train, X_test, y_all_train, y_all_test)
    jp_rfm = rfm.JPModel(X_train, X_test, y_all_train, y_all_test)
    print(f" Random Forest model accuracy for E-I: {accuracy_score(ei_rfm[1], ei_rfm[2]):.3f}")
    print(f" Random Forest model accuracy for N-S: {accuracy_score(ns_rfm[1], ns_rfm[2]):.3f}")
    print(f" Random Forest model accuracy for F-T: {accuracy_score(ft_rfm[1], ft_rfm[2]):.3f}")
    print(f" Random Forest model accuracy for J-P: {accuracy_score(jp_rfm[1], jp_rfm[2]):.3f}")