from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from Models import Models
from DataSet import DataSet

if __name__ == '__main__':
    lrm = Models(LogisticRegression(solver='lbfgs', random_state=1))
    rfm = Models(RandomForestClassifier(n_estimators=100, random_state=42))
    svm = Models(SVC())
    nbm = Models(MultinomialNB())

    dataset = DataSet()
    X_train, X_test, y_all_train, y_all_test, vectorizer = dataset.handle_data()

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

    ei_svm = svm.EIModel(X_train, X_test, y_all_train, y_all_test)
    ns_svm = svm.NSModel(X_train, X_test, y_all_train, y_all_test)
    ft_svm = svm.FTModel(X_train, X_test, y_all_train, y_all_test)
    jp_svm = svm.JPModel(X_train, X_test, y_all_train, y_all_test)
    print(f" SVM model accuracy for E-I: {accuracy_score(ei_svm[1], ei_svm[2]):.3f}")
    print(f" SVM model accuracy for N-S: {accuracy_score(ns_svm[1], ns_svm[2]):.3f}")
    print(f" SVM model accuracy for F-T: {accuracy_score(ft_svm[1], ft_svm[2]):.3f}")
    print(f" SVM model accuracy for J-P: {accuracy_score(jp_svm[1], jp_svm[2]):.3f}")

    ei_nbm = nbm.EIModel(X_train, X_test, y_all_train, y_all_test)
    ns_nbm = nbm.NSModel(X_train, X_test, y_all_train, y_all_test)
    ft_nbm = nbm.FTModel(X_train, X_test, y_all_train, y_all_test)
    jp_nbm = nbm.JPModel(X_train, X_test, y_all_train, y_all_test)
    print(f" Naive Bayes model accuracy for E-I: {accuracy_score(ei_nbm[1], ei_nbm[2]):.3f}")
    print(f" Naive Bayes model accuracy for N-S: {accuracy_score(ns_nbm[1], ns_nbm[2]):.3f}")
    print(f" Naive Bayes model accuracy for F-T: {accuracy_score(ft_nbm[1], ft_nbm[2]):.3f}")
    print(f" Naive Bayes model accuracy for J-P: {accuracy_score(jp_nbm[1], jp_nbm[2]):.3f}")
