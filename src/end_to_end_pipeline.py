import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def master_pipeline():
    print("=== STARTING END-TO-END NLP PIPELINE ===\n")

    # 1. Raw Data Ingestion
    raw_tickets = [
        "My Apple iPhone 14 is NOT turning on after the update in New York!",
        "Billing is wrong, you charged my Visa card twice.",
        "Can I upgrade my current internet plan to gigabit?"
    ]
    labels = ["Tech_Support", "Billing", "General"]

    print("1. DATA PREPROCESSING")
    cleaned_docs = [re.sub(r'[^a-zA-Z\s]', '', text).lower().strip() for text in raw_tickets]
    print(f"Cleaned Docs: {cleaned_docs}\n")

    # 2. Classification Module
    print("2. TEXT CLASSIFICATION (SVM)")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_docs)
    clf = LinearSVC()
    clf.fit(X, labels)
    print("Model Trained successfully on TF-IDF features.\n")

    # 3. NER Extraction Module
    print("3. NAMED ENTITY RECOGNITION")
    try:
        nlp = spacy.load("en_core_web_sm")
        for doc_text in raw_tickets:
            doc = nlp(doc_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            print(f"Entities in '{doc_text[:20]}...': {entities}")
    except OSError:
        print("spaCy model not loaded. Skipping NER.")
    print("\n")

    # 4. Advanced Analytics Simulation
    print("4. TOPIC MODELING & SUMMARIZATION (SIMULATED)")
    print("Extracted Themes: ['Hardware Failure', 'Payment Error', 'Plan Upgrade']")
    print("Summary: Multiple users reporting hardware and billing issues requiring immediate support.")
    print("\n=== PIPELINE EXECUTION COMPLETE ===")

if __name__ == "__main__":
    master_pipeline()