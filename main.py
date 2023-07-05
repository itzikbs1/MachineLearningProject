import re

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from DataSet import DataSet
from Models import Models

if __name__ == '__main__':
    dataset = DataSet()
    X_train, X_test, y_train, y_test, vectorizer = dataset.handle_data()
    # lrm = Models(LogisticRegression(solver='lbfgs', random_state=1))
    lrm = Models(SVC())
    pca = PCA(n_components=100)  # Specify the desired number of components
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    input_user = input("Please Enter text: ")
    replacements = [
        (r"(http.*?\s)", " "),
        (r"[^\w\s]", " "),
        (r"\_", " "),
        (r"\d+", " ")]

    for old, new in replacements:
        input_user = re.sub(old, new, input_user)

    input_user_Vectorized = vectorizer.transform([input_user])
    input_user_Vectorized = input_user_Vectorized.toarray()
    input_user_Vectorized = pca.transform(input_user_Vectorized)

    ei_lrm = lrm.EIModel(X_train, X_test, y_train, y_test)
    ei_prediction = ei_lrm[0].predict(input_user_Vectorized)

    ns_lrm = lrm.NSModel(X_train, X_test, y_train, y_test)
    ns_prediction = ns_lrm[0].predict(input_user_Vectorized)

    ft_lrm = lrm.FTModel(X_train, X_test, y_train, y_test)
    ft_prediction = ft_lrm[0].predict(input_user_Vectorized)

    jp_lrm = lrm.JPModel(X_train, X_test, y_train, y_test)
    jp_prediction = jp_lrm[0].predict(input_user_Vectorized)

    output_EI = 'E' if ei_prediction == 0 else "I"
    output_NS = 'N' if ns_prediction == 0 else "S"
    output_FT = 'F' if ft_prediction == 0 else "T"
    output_JP = 'J' if jp_prediction == 0 else "P"
    prediction_text = f'{output_EI}{output_NS}{output_FT}{output_JP}'

    print(prediction_text)
