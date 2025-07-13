# Step 1: Import Libraries and Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
# You can use any dataset; here we're using a common format for demonstration
# Format: ['label', 'message'], where label is 'spam' or 'ham'
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Step 2: Data Preprocessing
df.drop_duplicates(inplace=True)  # Remove duplicates
df.dropna(inplace=True)           # Remove nulls

# Convert labels to binary: 'spam' -> 1, 'ham' -> 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 3: Implement Models
# 1. Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# 2. Support Vector Machine (Linear SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)

# Step 4: Model Evaluation Function
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"\n📊 Evaluation for {model_name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

# Evaluate both models
evaluate_model(nb_model, X_test_vec, y_test, "Naive Bayes")
evaluate_model(svm_model, X_test_vec, y_test, "Support Vector Machine")

# Step 5: Testing with Custom Messages
sample_emails = [
    "Congratulations! You have won a $1000 Walmart gift card. Click to claim.",
    "Hi John, please send me the report by tomorrow.",
]

sample_features = vectorizer.transform(sample_emails)

print("\n🔍 Testing with custom samples:")
print("Naive Bayes Prediction:", nb_model.predict(sample_features))  # 1: spam, 0: ham
print("SVM Prediction:", svm_model.predict(sample_features))
