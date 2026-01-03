import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


def train_evaluate_mlp(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        alpha=1e-3,
        random_state=42,
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]

    print("=== MLP Results ===")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))

    return y_pred, y_prob


def train_evaluate_mnb(X_train, X_test, y_train, y_test):
    multinomialNB = MultinomialNB()
    multinomialNB.fit(X_train, y_train)

    y_pred = multinomialNB.predict(X_test)

    print("=== Multinomial Naive Bayes Results ===")
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision:", metrics.precision_score(y_test, y_pred))
    print("Recall:", metrics.recall_score(y_test, y_pred))
    print("F1:", metrics.f1_score(y_test, y_pred))

    return y_pred


def main():
    df = pd.read_csv("enron_spam_data.csv")

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    texts = df["Subject"].str.cat(df["Message"], sep=" ")

    vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)
    X = vectorizer.fit_transform(texts)

    y = df["Spam/Ham"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, random_state=42, stratify=y
    )

    # MLP
    y_pred_mlp, y_prob_mlp = train_evaluate_mlp(X_train, X_test, y_train, y_test)

    # Naive Bayes
    train_evaluate_mnb(X_train, X_test, y_train, y_test)

    # ROC Curve comparison for MLP and Naive Bayes

    # احتمال کلاس 1
    y_prob_mnb = mnb.predict_proba(X_test)[:, 1]

    fpr_mlp, tpr_mlp, _ = metrics.roc_curve(y_test, y_prob_mlp)
    auc_mlp = metrics.roc_auc_score(y_test, y_prob_mlp)

    fpr_nb, tpr_nb, _ = metrics.roc_curve(y_test, y_prob_mnb)
    auc_nb = metrics.roc_auc_score(y_test, y_prob_mnb)

    plt.figure()
    plt.plot(fpr_mlp, tpr_mlp, label=f"MLP (AUC = {auc_mlp:.3f})")
    plt.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC = {auc_nb:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")  # خط تصادفی
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
