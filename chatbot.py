import json
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load FAQ data
with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)

questions = [item["question"] for item in faqs]
answers = [item["answer"] for item in faqs]

# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words="english")
faq_vectors = vectorizer.fit_transform(questions)


def get_answer(user_question):
    if not user_question.strip():
        return "Please enter a question."

    # Convert user's question into a vector
    user_vector = vectorizer.transform([user_question])

    # Calculate similarity
    similarities = cosine_similarity(user_vector, faq_vectors)[0]

    # Find the most similar question
    best_index = similarities.argmax()
    best_score = similarities[best_index]

    # Confidence threshold
    if best_score < 0.20:
        return "Sorry, I could not find a suitable answer. Please try asking in a different way."

    return answers[best_index]


def send_message():
    question = entry.get().strip()

    if not question:
        return

    chat_area.config(state="normal")
    chat_area.insert(tk.END, "You: " + question + "\n")
    chat_area.insert(tk.END, "Bot: " + get_answer(question) + "\n\n")
    chat_area.config(state="disabled")
    chat_area.see(tk.END)

    entry.delete(0, tk.END)


# Create chatbot window
root = tk.Tk()
root.title("FAQ Chatbot")
root.geometry("650x600")

title = tk.Label(
    root,
    text="FAQ Chatbot",
    font=("Arial", 20, "bold")
)
title.pack(pady=10)

subtitle = tk.Label(
    root,
    text="Ask a question about college services",
    font=("Arial", 11)
)
subtitle.pack()

# Chat area
chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 11),
    state="disabled"
)
chat_area.pack(
    padx=15,
    pady=15,
    fill=tk.BOTH,
    expand=True
)

# Input area
entry_frame = tk.Frame(root)
entry_frame.pack(
    fill=tk.X,
    padx=15,
    pady=10
)

entry = tk.Entry(
    entry_frame,
    font=("Arial", 12)
)
entry.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    ipady=8
)

send_button = tk.Button(
    entry_frame,
    text="Send",
    font=("Arial", 11, "bold"),
    command=send_message
)
send_button.pack(
    side=tk.RIGHT,
    padx=(8, 0)
)

# Welcome message
chat_area.config(state="normal")
chat_area.insert(
    tk.END,
    "Bot: Hello! I am your FAQ chatbot.\n"
    "Ask me something about college services.\n\n"
)
chat_area.config(state="disabled")

root.mainloop()