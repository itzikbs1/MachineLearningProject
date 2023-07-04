import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score


class Models:

    def __init__(self, classifier):
        self.classifier = classifier  # create the classifier for each class

    def EIModel(self, X_train, X_test, y_all_train, y_all_test):
        # split the data for E-I
        y_EI_train = y_all_train['E0-I1']
        y_EI_test = y_all_test['E0-I1']
        self.classifier.fit(X_train, y_EI_train)
        y_EI_pred = self.classifier.predict(X_test)
        return self.classifier, y_EI_test, y_EI_pred

    def NSModel(self, X_train, X_test, y_all_train, y_all_test):
        # split the data for N-S
        y_NS_train = y_all_train['N0-S1']
        y_NS_test = y_all_test['N0-S1']

        self.classifier.fit(X_train, y_NS_train)

        y_NS_pred = self.classifier.predict(X_test)
        return self.classifier, y_NS_test, y_NS_pred

    def FTModel(self, X_train, X_test, y_all_train, y_all_test):
        # split the data for F-T
        y_FT_train = y_all_train['F0-T1']
        y_FT_test = y_all_test['F0-T1']

        self.classifier.fit(X_train, y_FT_train)
        y_FT_pred = self.classifier.predict(X_test)
        return self.classifier, y_FT_test, y_FT_pred

    def JPModel(self, X_train, X_test, y_all_train, y_all_test):
        # split the data for J-P
        y_JP_train = y_all_train['J0-P1']
        y_JP_test = y_all_test['J0-P1']

        self.classifier.fit(X_train, y_JP_train)

        y_JP_pred = self.classifier.predict(X_test)

        return self.classifier, y_JP_test, y_JP_pred
