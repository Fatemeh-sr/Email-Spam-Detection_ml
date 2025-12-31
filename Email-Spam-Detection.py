import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.naive_bayes import MultinomialNB


def main():
    df = pd.read_csv("enron_spam_data.csv")
    df.info()

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    print(df.isnull().sum())
    # print(df)

    # vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(
        stop_words="english", max_features=15000
    )  # max_features=8000,

    # df["text"] = df["Subject"] + " " + df["Message"]
    # X = vectorizer.fit_transform(df["text"])

    texts = df["Subject"].str.cat(df["Message"], sep=" ")
    X = vectorizer.fit_transform(texts)
    print("TF-IDF shape:", X.shape)  # 33,716 ایمیل     n ویژگی (کلمه مهم)

    # y = df["Spam/Ham"].replace({"ham": 0, "spam": 1})

    y = df["Spam/Ham"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y
    )

    # mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=1e-3,
        random_state=42,
    )

    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)  # pish biniiiii
    y_prob_mlp = mlp.predict_proba(X_test)[:, 1]

    # moghayese y test , y pred

    print("MLP Accuracy:", accuracy_score(y_test, y_pred_mlp))
    print("MLP Precision:", precision_score(y_test, y_pred_mlp))
    print("MLP Recall:", recall_score(y_test, y_pred_mlp))
    print("MLP F1:", f1_score(y_test, y_pred_mlp))

    print("\nClassification Report (MLP):")
    print(classification_report(y_test, y_pred_mlp))

    fpr, tpr, _ = roc_curve(y_test, y_prob_mlp)
    auc_mlp = roc_auc_score(y_test, y_prob_mlp)

    plt.figure()
    plt.plot(fpr, tpr, label=f"MLP (AUC = {auc_mlp:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - MLP")
    plt.legend()
    plt.show()
    y_train_pred = mlp.predict(X_train)

    print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_pred_mlp))

    multinomialNB = MultinomialNB()
    multinomialNB.fit(X_train, y_train)
    y_pred_MNB = multinomialNB.predict(X_test)

    print("MultinomialNB Accuracy:", accuracy_score(y_test, y_pred_MNB))
    print("MultinomialNB Precision:", precision_score(y_test, y_pred_MNB))
    print("MultinomialNB Recall:", recall_score(y_test, y_pred_MNB))
    print("MultinomialNB F1:", f1_score(y_test, y_pred_MNB))


# درست کردن پرینت ها و متریک

if __name__ == "__main__":
    main()
