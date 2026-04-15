"""Logistic Regression baseline with TF-IDF features."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from classifier.baselines.common import load_data


def run_logistic_regression(train_df, test_df, le):
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION (TF-IDF)")
    print("=" * 50)

    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    X_train = tfidf.fit_transform(train_df["clean"])
    X_test = tfidf.transform(test_df["clean"])

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro-F1: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    train_df, test_df, le = load_data()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}  Classes: {list(le.classes_)}")
    run_logistic_regression(train_df, test_df, le)
