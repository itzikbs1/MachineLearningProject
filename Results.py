import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from Models import Models
from DataSet import DataSet
import time
import pandas as pd

if __name__ == '__main__':
    lrm = Models(LogisticRegression(solver='lbfgs', random_state=1))
    rfm = Models(RandomForestClassifier(n_estimators=100, random_state=42))
    svm = Models(SVC())
    nbm = Models(MultinomialNB())
    ada = Models(AdaBoostClassifier(n_estimators=50, random_state=42))
    knn = Models(
        KNeighborsClassifier(n_neighbors=82))  # 82 neighbors gave the best results after checking 100 neighbors

    dataset = DataSet()
    X_train, X_test, y_train, y_test, vectorizer = dataset.handle_data()

    start_lrm = time.time()
    ei_lrm = lrm.EIModel(X_train, X_test, y_train, y_test)
    ns_lrm = lrm.NSModel(X_train, X_test, y_train, y_test)
    ft_lrm = lrm.FTModel(X_train, X_test, y_train, y_test)
    jp_lrm = lrm.JPModel(X_train, X_test, y_train, y_test)
    print(f" Logistic regression model accuracy for E-I: {accuracy_score(ei_lrm[1], ei_lrm[2]):.3f}")
    print(f" Logistic regression model accuracy for N-S: {accuracy_score(ns_lrm[1], ns_lrm[2]):.3f}")
    print(f" Logistic regression model accuracy for F-T: {accuracy_score(ft_lrm[1], ft_lrm[2]):.3f}")
    print(f" Logistic regression model accuracy for J-P: {accuracy_score(jp_lrm[1], jp_lrm[2]):.3f}")
    end_lrm = time.time()
    print(f"Time for Logistic regression model:{end_lrm - start_lrm}")

    start_rfm = time.time()
    ei_rfm = rfm.EIModel(X_train, X_test, y_train, y_test)
    ns_rfm = rfm.NSModel(X_train, X_test, y_train, y_test)
    ft_rfm = rfm.FTModel(X_train, X_test, y_train, y_test)
    jp_rfm = rfm.JPModel(X_train, X_test, y_train, y_test)
    print(f" Random Forest model accuracy for E-I: {accuracy_score(ei_rfm[1], ei_rfm[2]):.3f}")
    print(f" Random Forest model accuracy for N-S: {accuracy_score(ns_rfm[1], ns_rfm[2]):.3f}")
    print(f" Random Forest model accuracy for F-T: {accuracy_score(ft_rfm[1], ft_rfm[2]):.3f}")
    print(f" Random Forest model accuracy for J-P: {accuracy_score(jp_rfm[1], jp_rfm[2]):.3f}")
    end_rfm = time.time()
    print(f"Time for Random Forest model:{end_rfm - start_rfm}")

    start_svm = time.time()
    # Create your classification models and train them using the PCA-transformed data
    pca = PCA(n_components=20)  # Specify the desired number of components
    X_svm_train = X_train.toarray()
    X__svm_test = X_test.toarray()
    X_train_pca = pca.fit_transform(X_svm_train)
    X_test_pca = pca.transform(X__svm_test)

    ei_svm = svm.EIModel(X_train_pca, X_test_pca, y_train, y_test)
    ns_svm = svm.NSModel(X_train_pca, X_test_pca, y_train, y_test)
    ft_svm = svm.FTModel(X_train_pca, X_test_pca, y_train, y_test)
    jp_svm = svm.JPModel(X_train_pca, X_test_pca, y_train, y_test)
    # ei_svm = svm.EIModel(X_train, X_test, y_train, y_test)
    # ns_svm = svm.NSModel(X_train, X_test, y_train, y_test)
    # ft_svm = svm.FTModel(X_train, X_test, y_train, y_test)
    # jp_svm = svm.JPModel(X_train, X_test, y_train, y_test)
    print(f" SVM model accuracy for E-I: {accuracy_score(ei_svm[1], ei_svm[2]):.3f}")
    print(f" SVM model accuracy for N-S: {accuracy_score(ns_svm[1], ns_svm[2]):.3f}")
    print(f" SVM model accuracy for F-T: {accuracy_score(ft_svm[1], ft_svm[2]):.3f}")
    print(f" SVM model accuracy for J-P: {accuracy_score(jp_svm[1], jp_svm[2]):.3f}")
    end_svm = time.time()
    print(f"Time for SVM model:{end_svm - start_svm}")

    start_nbm = time.time()
    ei_nbm = nbm.EIModel(X_train, X_test, y_train, y_test)
    ns_nbm = nbm.NSModel(X_train, X_test, y_train, y_test)
    ft_nbm = nbm.FTModel(X_train, X_test, y_train, y_test)
    jp_nbm = nbm.JPModel(X_train, X_test, y_train, y_test)
    print(f" Naive Bayes model accuracy for E-I: {accuracy_score(ei_nbm[1], ei_nbm[2]):.3f}")
    print(f" Naive Bayes model accuracy for N-S: {accuracy_score(ns_nbm[1], ns_nbm[2]):.3f}")
    print(f" Naive Bayes model accuracy for F-T: {accuracy_score(ft_nbm[1], ft_nbm[2]):.3f}")
    print(f" Naive Bayes model accuracy for J-P: {accuracy_score(jp_nbm[1], jp_nbm[2]):.3f}")
    end_nbm = time.time()
    print(f"Time for Naive Bayes model:{end_nbm - start_nbm}")

    start_ada = time.time()
    ei_ada = ada.EIModel(X_train, X_test, y_train, y_test)
    ns_ada = ada.NSModel(X_train, X_test, y_train, y_test)
    ft_ada = ada.FTModel(X_train, X_test, y_train, y_test)
    jp_ada = ada.JPModel(X_train, X_test, y_train, y_test)
    print(f" AdaBoost model accuracy for E-I: {accuracy_score(ei_ada[1], ei_ada[2]):.3f}")
    print(f" AdaBoost model accuracy for N-S: {accuracy_score(ns_ada[1], ns_ada[2]):.3f}")
    print(f" AdaBoost model accuracy for F-T: {accuracy_score(ft_ada[1], ft_ada[2]):.3f}")
    print(f" AdaBoost model accuracy for J-P: {accuracy_score(jp_ada[1], jp_ada[2]):.3f}")
    end_ada = time.time()
    print(f"Time for AdaBoost model:{end_ada - start_ada}")
    start_knn = time.time()
    ei_knn = knn.EIModel(X_train, X_test, y_train, y_test)
    ns_knn = knn.NSModel(X_train, X_test, y_train, y_test)
    ft_knn = knn.FTModel(X_train, X_test, y_train, y_test)
    jp_knn = knn.JPModel(X_train, X_test, y_train, y_test)
    print(f" KNeighbors model accuracy for E-I: {accuracy_score(ei_knn[1], ei_knn[2]):.3f}")
    print(f" KNeighbors model accuracy for N-S: {accuracy_score(ns_knn[1], ns_knn[2]):.3f}")
    print(f" KNeighbors model accuracy for F-T: {accuracy_score(ft_knn[1], ft_knn[2]):.3f}")
    print(f" KNeighbors model accuracy for J-P: {accuracy_score(jp_knn[1], jp_knn[2]):.3f}")
    end_knn = time.time()
    print(f"Time for Logistic regression model:{end_knn - start_knn}")

    classifiers = ['Logistic Regression', 'Random Forest', 'SVM', 'Multinomial NB', 'AdaBoost', 'KNeighbors']

    # Accuracy scores for each classifier
    accuracy_scores_lrm = [accuracy_score(ei_lrm[1], ei_lrm[2]), accuracy_score(ns_lrm[1], ns_lrm[2]),
                           accuracy_score(ft_lrm[1], ft_lrm[2]), accuracy_score(jp_lrm[1], jp_lrm[2])]

    accuracy_scores_rfc = [accuracy_score(ei_rfm[1], ei_rfm[2]), accuracy_score(ns_rfm[1], ns_rfm[2]),
                           accuracy_score(ft_rfm[1], ft_rfm[2]), accuracy_score(jp_rfm[1], jp_rfm[2])]

    accuracy_scores_svm = [accuracy_score(ei_svm[1], ei_svm[2]), accuracy_score(ns_svm[1], ns_svm[2]),
                           accuracy_score(ft_svm[1], ft_svm[2]), accuracy_score(jp_svm[1], jp_svm[2])]

    accuracy_scores_nbm = [accuracy_score(ei_nbm[1], ei_nbm[2]), accuracy_score(ns_nbm[1], ns_nbm[2]),
                           accuracy_score(ft_nbm[1], ft_nbm[2]), accuracy_score(jp_nbm[1], jp_nbm[2])]

    accuracy_scores_ada = [accuracy_score(ei_ada[1], ei_ada[2]), accuracy_score(ns_ada[1], ns_ada[2]),
                           accuracy_score(ft_ada[1], ft_ada[2]), accuracy_score(jp_ada[1], jp_ada[2])]

    accuracy_scores_knn = [accuracy_score(ei_knn[1], ei_knn[2]), accuracy_score(ns_knn[1], ns_knn[2]),
                           accuracy_score(ft_knn[1], ft_knn[2]), accuracy_score(jp_knn[1], jp_knn[2])]

    x_labels = ['E-I', 'N-S', 'F-T', 'J-P']
    bar_width = 0.2
    indices = np.arange(len(x_labels))

    plt.bar(indices, accuracy_scores_lrm, bar_width, label='Logistic Regression')
    plt.bar(indices + bar_width, accuracy_scores_rfc, bar_width, label='Random Forest')
    plt.bar(indices + 2 * bar_width, accuracy_scores_nbm, bar_width, label='Multinomial NB')
    plt.bar(indices + 3 * bar_width, accuracy_scores_svm, bar_width, label='SVM')
    plt.bar(indices + 4 * bar_width, accuracy_scores_ada, bar_width, label='AdaBoost')
    plt.bar(indices + 5 * bar_width, accuracy_scores_knn, bar_width, label='KNeighbors')
    plt.xticks(indices + bar_width, x_labels)
    plt.xlabel('Results')
    plt.ylabel('Accuracies')
    plt.title('Results for Different Classifiers')
    plt.legend()
    plt.show()
