import csv
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))


def transform_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return " ".join(words)


def load_data(path):
    texts = []
    labels = []
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get('v1')
            text = row.get('v2')
            if label is None or text is None:
                continue
            labels.append(1 if label.strip().lower() == 'spam' else 0)
            texts.append(transform_text(text))
    return texts, labels


def main():
    texts, labels = load_data('spam.csv')
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification report:')
    print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Saved vectorizer.pkl and model.pkl')


if __name__ == '__main__':
    main()
