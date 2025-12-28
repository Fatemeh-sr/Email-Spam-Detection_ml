import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    df = pd.read_csv("enron_spam_data.csv")
    df.info()

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    print(df.isnull().sum())
    # print(df)

    # vectorizer = TfidfVectorizer()
    vectorizer = TfidfVectorizer(stop_words="english")  # max_features=8000,

    # df["text"] = df["Subject"] + " " + df["Message"]
    # X = vectorizer.fit_transform(df["text"])

    texts = df["Subject"].str.cat(df["Message"], sep=" ")
    X = vectorizer.fit_transform(texts)
    print("TF-IDF shape:", X.shape)  # 33,716 ایمیل     n ویژگی (کلمه مهم)


if __name__ == "__main__":
    main()
