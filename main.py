import pandas as pd
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load your dataset
 
def load_dataset(file_path="C:\\Users\\monis\\OneDrive\\Desktop\\batch-11\\Book2.csv"):
    return pd.read_csv(r"C:\Users\monis\OneDrive\Desktop\batch-11\Book2.csv")

# Function to preprocess the message
def preprocess_message(message):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(message.lower())
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(filtered_tokens)

# Function to train a classifier
def train_classifier(dataset):
    processed_messages = [(preprocess_message(message), label) for message, label in zip(dataset['message'], dataset['label'])]

    X = [message for message, _ in processed_messages]
    y = [label for _, label in processed_messages]

    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(X)

    classifier = MultinomialNB()
    classifier.fit(X_tfidf, y)

    return classifier, tfidf_vectorizer

# Function to classify a message
def classify_message(message, classifier, tfidf_vectorizer):
    processed_message = preprocess_message(message)
    tfidf_message = tfidf_vectorizer.transform([processed_message])
    prediction = classifier.predict(tfidf_message)
    return prediction[0]

# Simple chatbot
def chat(dataset, classifier, tfidf_vectorizer):
    print("Hello! I'm your chatbot. You can ask me anything.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        elif user_input.lower() in ['hi', 'hello']:
            print(random.choice(["Hi there!", "Hello!", "Hey!"]))
        else:
            label = check_message(dataset, user_input)
            if label is not None:
                print(f"Chatbot: This message exists in the dataset and its label is {label}.")
            else:
                feedback = input("Chatbot: This message does not exist in the dataset. Is it spam? (yes/no) ").strip().lower()
                if feedback == 'yes':
                    new_label = 'spam'
                else:
                    new_label = 'ham'
                update_dataset(dataset, user_input, new_label)
                print("Chatbot: Thank you for your feedback. The dataset has been updated.")
                
# Function to check if a message exists in the dataset and return its label
def check_message(dataset, message):
    message_exists = message.lower() in dataset['message'].str.lower().values
    if message_exists:
        label = dataset.loc[dataset['message'].str.lower() == message.lower(), 'label'].iloc[0]
        return label
    else:
        return None

# Function to update the dataset with user feedback
def update_dataset(dataset, message, label):
    new_row = pd.DataFrame({'message': [message], 'label': [label]})
    dataset = pd.concat([dataset, new_row], ignore_index=True)
    dataset.to_csv(r"C:\Users\monis\OneDrive\Desktop\batch-11\Book2.csv", index=False)

# Load the dataset
dataset = load_dataset(r"C:\Users\monis\OneDrive\Desktop\batch-11\Book2.csv")

# Train the classifier
classifier, tfidf_vectorizer = train_classifier(dataset)

# Start the chatbot
chat(dataset, classifier, tfidf_vectorizer)