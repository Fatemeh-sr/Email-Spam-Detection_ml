import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


def train_MLPClassifier(X_train, X_test, y_train, y_test):
    model = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=1e-3,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)

    print("------------- MLP -------------\n")
    print("Accuracy:", metrics.accuracy_score(y_test, predicted_y))
    print("Precision:", metrics.precision_score(y_test, predicted_y))
    print("Recall:", metrics.recall_score(y_test, predicted_y))
    print("F1:", metrics.f1_score(y_test, predicted_y))
    print(metrics.classification_report(y_test, predicted_y))

    return model


def train_MultinomialNB(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)

    predicted_y = model.predict(X_test)

    print("------------- Multinomial Naive Bayes -------------\n")
    print("Accuracy:", metrics.accuracy_score(y_test, predicted_y))
    print("Precision:", metrics.precision_score(y_test, predicted_y))
    print("Recall:", metrics.recall_score(y_test, predicted_y))
    print("F1:", metrics.f1_score(y_test, predicted_y))
    print(metrics.classification_report(y_test, predicted_y))

    return model


def predict_new_email(this_file, vectorizer, mlp_model, mnb_model):
    try:
        with open(this_file, encoding="utf-8") as my_file:
            email_text = my_file.read()

        email_vector = vectorizer.transform([email_text])

        mlp_pred = mlp_model.predict(email_vector)[0]
        mlp_label = "spam" if mlp_pred == 1 else "ham"
        print(f"MLP Prediction: {mlp_label}")

        mnb_pred = mnb_model.predict(email_vector)[0]
        mnb_label = "spam" if mnb_pred == 1 else "ham"
        print(f"Naive Bayes Prediction: {mnb_label}")

    except FileNotFoundError:
        print(f"File '{this_file}' not found.")
    except Exception as e:
        print("Error:", e)


def main():
    df = pd.read_csv("enron_spam_data.csv")

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    texts = df["Subject"].str.cat(df["Message"], sep=" ")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)

    y = df["Spam/Ham"].map({"ham": 0, "spam": 1})

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts, y, train_size=0.7, random_state=42, stratify=y
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # MLP
    # Naive Bayes
    mlp_model = train_MLPClassifier(X_train, X_test, y_train, y_test)
    mnb_model = train_MultinomialNB(X_train, X_test, y_train, y_test)

    # predict_new_email("new_email.txt", vectorizer, mlp_model, mnb_model)


if __name__ == "__main__":
    main()
