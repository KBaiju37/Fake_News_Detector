import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

true_df = pd.read_csv(r"C:\Users\Baijnath\Downloads\archive_NLP\True.csv")
fake_df = pd.read_csv(r"C:\Users\Baijnath\Downloads\archive_NLP\Fake.csv")

true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

df = pd.concat([true_df, fake_df])
df = df[['title', 'text', 'label']]
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Confusion Matrix:")
print(matrix)
print("Classification Report:")
print(report)
