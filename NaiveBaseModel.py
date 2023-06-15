import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split


class NaiveBaseModel:

    def __init__(self):
        # create log reg model
        self.classifier = GaussianNB()

        # implement random oversampling
        self.ros = RandomOverSampler(random_state=1)

    def EIModel(self, X_train, X_test, y_all_train, y_all_test):
        y_EI_train = y_all_train['E0-I1']
        y_EI_test = y_all_test['E0-I1']

        X_resampled_ros, y_EI_resampled_ros = self.ros.fit_resample(X_train, y_EI_train)
        # Fit E-I combination with oversampled x_train and y_EI_train
        self.classifier.fit(X_resampled_ros, y_EI_resampled_ros)
        # Predict outcomes for test data set
        y_EI_pred_ros = self.classifier.predict(X_test)
        # EI_result = pd.DataFrame({"Prediction": y_EI_pred_ros, "Actual": y_EI_test})
        return self.classifier, y_EI_test, y_EI_pred_ros

    def NSModel(self, X_train, X_test, y_all_train, y_all_test):
        # resample N-S combination
        y_NS_train = y_all_train['N0-S1']
        y_NS_test = y_all_test['N0-S1']

        X_resampled_ros, y_NS_resampled_ros = self.ros.fit_resample(X_train, y_NS_train)

        self.classifier.fit(X_resampled_ros, y_NS_resampled_ros)

        # Predict outcomes for test data set
        y_NS_pred_ros = self.classifier.predict(X_test)
        # NS_result = pd.DataFrame({"Prediction": y_NS_pred_ros, "Actual": y_NS_test})
        # NS_result.head(5)
        return self.classifier, y_NS_test, y_NS_pred_ros

    def FTModel(self, X_train, X_test, y_all_train, y_all_test):
        # resample F-T combination
        y_FT_train = y_all_train['F0-T1']
        y_FT_test = y_all_test['F0-T1']

        X_resampled_ros, y_FT_resampled_ros = self.ros.fit_resample(X_train, y_FT_train)
        # Fit F-T combination with oversampled x_train and y_FT_train
        self.classifier.fit(X_resampled_ros, y_FT_resampled_ros)
        # Predict outcomes for test data set
        y_FT_pred_ros = self.classifier.predict(X_test)
        # FT_result = pd.DataFrame({"Prediction": y_FT_pred_ros, "Actual": y_FT_test})
        # FT_result.head(5)
        # resample J-P combination
        return self.classifier, y_FT_test, y_FT_pred_ros

    def JPModel(self, X_train, X_test, y_all_train, y_all_test):
        y_JP_train = y_all_train['J0-P1']
        y_JP_test = y_all_test['J0-P1']

        X_resampled_ros, y_JP_resampled_ros = self.ros.fit_resample(X_train, y_JP_train)
        # Fit J-P combination with oversampled x_train and y_JP_train
        self.classifier.fit(X_resampled_ros, y_JP_resampled_ros)

        # Predict outcomes for test data set
        y_JP_pred_ros = self.classifier.predict(X_test)
        # JP_result = pd.DataFrame({"Prediction": y_JP_pred_ros, "Actual": y_JP_test})

        return self.classifier, y_JP_test, y_JP_pred_ros
        # JP_result.head(5)
