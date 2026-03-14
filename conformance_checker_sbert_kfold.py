############################################################
# HYBRID REQUIREMENT CONFORMANCE CHECKER
# ----------------------------------------------------------
# This script performs:
#   - CSV-based dataset loading (negation.csv, voice.csv, superlatives.csv)
#   - SBERT embeddings (all-MiniLM-L6-v2)
#   - Logistic Regression classifiers (negation + voice + superlatives)
#   - Grammar-based passive voice detection (spaCy)
#   - Hybrid ML + grammar decision with confidence fallback (voice only)
#   - Hybrid ML + lexical + linguistic + quantifier detection (superlatives)
#   - K-fold evaluation for all ML classifiers
#   - Combined disagreement logging (voice + superlatives)
#   - Interactive conformance checking loop
############################################################

# -----------------------------
# IMPORTS
# -----------------------------

import os
import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import spacy


# -----------------------------
# MODEL LOADING
# -----------------------------

# Load SBERT model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load spaCy English model (user must install manually)
nlp = spacy.load("en_core_web_sm")


# -----------------------------
# DATA LOADING
# -----------------------------

def load_training_data_csv(neg_csv="negation.csv",
                           voice_csv="voice.csv",
                           super_csv="superlatives.csv"):
    """
    Loads negation, voice, and superlatives datasets from CSV files.
    Each CSV must contain: text,label
    """

    for path in [neg_csv, voice_csv, super_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    neg_df = pd.read_csv(neg_csv, encoding="utf-8-sig")
    voice_df = pd.read_csv(voice_csv, encoding="utf-8-sig")
    super_df = pd.read_csv(super_csv, encoding="utf-8-sig")

    # Clean column names
    neg_df.columns = neg_df.columns.str.strip()
    voice_df.columns = voice_df.columns.str.strip()
    super_df.columns = super_df.columns.str.strip()

    return neg_df, voice_df, super_df


# -----------------------------
# GRAMMAR-BASED PASSIVE DETECTION
# -----------------------------

def is_passive_grammar(text: str) -> bool:
    """
    Uses spaCy dependency parsing to detect passive voice.
    Passive voice typically contains:
        - auxpass (passive auxiliary)
        - nsubjpass (passive subject)
    """

    doc = nlp(text)

    for token in doc:
        if token.dep_ in ("auxpass", "nsubjpass"):
            return True

    return False


# -----------------------------
# K-FOLD EVALUATION (ML ONLY)
# -----------------------------

def evaluate_classifier_kfold(embeddings, labels, k=5):
    """
    Performs k-fold cross-validation using Logistic Regression.
    Returns average accuracy, precision, recall, and F1 score.
    """

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_idx, test_idx in kf.split(embeddings):

        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=200)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)

        accuracies.append(accuracy_score(y_test, preds))
        precisions.append(precision_score(y_test, preds, zero_division=0))
        recalls.append(recall_score(y_test, preds, zero_division=0))
        f1s.append(f1_score(y_test, preds, zero_division=0))

    return {
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s),
    }


def run_kfold_evaluation(k=5):
    """
    Loads CSV datasets, encodes them with SBERT,
    and runs k-fold evaluation for all three classifiers.
    """

    neg_df, voice_df, super_df = load_training_data_csv()

    # NEGATION
    X_neg = model.encode(neg_df["text"].astype(str).tolist())
    y_neg = neg_df["label"].astype(int).values
    neg_results = evaluate_classifier_kfold(X_neg, y_neg, k=k)

    # VOICE
    X_voice = model.encode(voice_df["text"].astype(str).tolist())
    y_voice = voice_df["label"].astype(int).values
    voice_results = evaluate_classifier_kfold(X_voice, y_voice, k=k)

    # SUPERLATIVES
    X_super = model.encode(super_df["text"].astype(str).tolist())
    y_super = super_df["label"].astype(int).values
    super_results = evaluate_classifier_kfold(X_super, y_super, k=k)

    print("\n=== K-FOLD EVALUATION RESULTS ===")

    print("\nNEGATION CLASSIFIER:")
    for metric, value in neg_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nVOICE CLASSIFIER:")
    for metric, value in voice_results.items():
        print(f"{metric}: {value:.4f}")

    print("\nSUPERLATIVES CLASSIFIER:")
    for metric, value in super_results.items():
        print(f"{metric}: {value:.4f}")


# -----------------------------
# TRAIN FINAL ML CLASSIFIERS
# -----------------------------

def train_classifiers_csv():
    """
    Trains final negation, voice, and superlatives classifiers.
    Saves them as .pkl files.
    """

    neg_df, voice_df, super_df = load_training_data_csv()

    # NEGATION
    X_neg = model.encode(neg_df["text"].astype(str).tolist())
    y_neg = neg_df["label"].astype(int).values
    neg_clf = LogisticRegression(max_iter=200)
    neg_clf.fit(X_neg, y_neg)
    joblib.dump(neg_clf, "negation_classifier.pkl")

    # VOICE
    X_voice = model.encode(voice_df["text"].astype(str).tolist())
    y_voice = voice_df["label"].astype(int).values
    voice_clf = LogisticRegression(max_iter=200)
    voice_clf.fit(X_voice, y_voice)
    joblib.dump(voice_clf, "voice_classifier.pkl")

    # SUPERLATIVES
    X_super = model.encode(super_df["text"].astype(str).tolist())
    y_super = super_df["label"].astype(int).values
    super_clf = LogisticRegression(max_iter=200)
    super_clf.fit(X_super, y_super)
    joblib.dump(super_clf, "superlatives_classifier.pkl")

    print("\nFinal ML classifiers trained and saved.")


def load_classifiers():
    """
    Loads trained ML classifiers from disk.
    """

    neg_clf = joblib.load("negation_classifier.pkl")
    voice_clf = joblib.load("voice_classifier.pkl")
    super_clf = joblib.load("superlatives_classifier.pkl")

    return neg_clf, voice_clf, super_clf


# ============================================================
# HYBRID VOICE DECISION (S‑B: line‑by‑line comments)
# ============================================================

def hybrid_voice_decision(text: str, voice_clf, disagreement_log: list, conf_threshold: float = 0.6):
    """
    Hybrid passive voice detection:
        - ML probability
        - Grammar-based detection
        - Confidence fallback
        - Logging disagreements
    """

    # Encode the text using SBERT
    emb = model.encode([text])

    # Get ML probability distribution for passive vs active
    proba = voice_clf.predict_proba(emb)[0]

    # Probability that the sentence is passive
    p_passive = proba[1]

    # ML decision: passive if probability >= 0.5
    ml_passive = p_passive >= 0.5

    # Grammar-based passive detection using spaCy
    grammar_passive = is_passive_grammar(text)

    # If ML is confident (above threshold or below inverse threshold)
    if p_passive >= conf_threshold or p_passive <= (1 - conf_threshold):
        # Use ML decision
        final_passive = ml_passive
        source = "ML"
    else:
        # Otherwise fall back to grammar rule
        final_passive = grammar_passive
        source = "GRAMMAR"

    # Log disagreement if ML and grammar disagree
    if ml_passive != grammar_passive:
        disagreement_log.append({
            "text": text,
            "type": "voice",
            "ml_passive": ml_passive,
            "grammar_passive": grammar_passive,
            "p_passive": p_passive,
            "final_passive": final_passive,
            "decision_source": source,
        })

    return final_passive, p_passive, source


# ============================================================
# SUPERLATIVE DETECTORS (S‑B: line‑by‑line comments)
# ============================================================

# -----------------------------
# LEXICAL SUPERLATIVE DETECTOR
# -----------------------------

LEXICAL_SUPERLATIVES = {
    "best", "worst", "maximum", "minimum", "optimal",
    "topmost", "foremost", "paramount", "unmatched",
    "unparalleled", "superior", "supreme"
}

def lexical_superlative_detector(text: str) -> bool:
    """
    Detects lexical superlatives using a simple word match.
    """

    # Convert text to lowercase for case-insensitive matching
    t = text.lower()

    # Split into tokens
    tokens = t.split()

    # Check if any token is in the lexical superlative list
    for tok in tokens:
        if tok in LEXICAL_SUPERLATIVES:
            return True

    return False


# -----------------------------
# LINGUISTIC SUPERLATIVE DETECTOR
# -----------------------------

def linguistic_superlative_detector(text: str) -> bool:
    """
    Detects linguistic superlatives using spaCy POS tags:
        - JJS (superlative adjectives)
        - RBS (superlative adverbs)
        - "most + ADJ"
        - "least + ADJ"
    """

    doc = nlp(text)

    for i, token in enumerate(doc):

        # JJS = superlative adjective (fastest, slowest, highest)
        if token.tag_ == "JJS":
            return True

        # RBS = superlative adverb (most efficiently, least reliably)
        if token.tag_ == "RBS":
            return True

        # "most + ADJ"
        if token.lower_ == "most" and i + 1 < len(doc):
            if doc[i + 1].pos_ == "ADJ":
                return True

        # "least + ADJ"
        if token.lower_ == "least" and i + 1 < len(doc):
            if doc[i + 1].pos_ == "ADJ":
                return True

    return False


# -----------------------------
# QUANTIFIER SUPERLATIVE DETECTOR (QF3)
# -----------------------------

def quantifier_superlative_detector(text: str) -> bool:
    """
    Detects quantifier superlatives:
        - "most" followed by a noun
        - "least" followed by a noun
        - "most of"
        - "least of"
        - Hyphenated forms: most-used, least-tested
    """

    doc = nlp(text)

    for i, token in enumerate(doc):

        # Hyphenated forms: most-used, least-tested
        if re.match(r"^(most|least)-", token.text.lower()):
            return True

        # "most" or "least" as separate tokens
        if token.lower_ in ("most", "least"):

            # Case 1: "most of"
            if i + 1 < len(doc) and doc[i + 1].lower_ == "of":
                return True

            # Case 2: "most" followed by a noun
            if i + 1 < len(doc) and doc[i + 1].pos_ in ("NOUN", "PROPN"):
                return True

    return False


# ============================================================
# HYBRID SUPERLATIVE DECISION (S‑B: line‑by‑line comments)
# ============================================================

def hybrid_superlative_decision(text: str, super_clf, disagreement_log: list):
    """
    Hybrid superlative detection:
        - ML classifier
        - Lexical rule
        - Linguistic rule
        - Quantifier rule
        - Combined OR fusion
        - Logging disagreements
    """

    # Encode text for ML classifier
    emb = model.encode([text])

    # ML prediction (0 or 1)
    ml_pred = int(super_clf.predict(emb)[0])

    # Lexical rule detection
    lex = lexical_superlative_detector(text)

    # Linguistic rule detection
    ling = linguistic_superlative_detector(text)

    # Quantifier rule detection
    quant = quantifier_superlative_detector(text)

    # Final OR fusion
    final = ml_pred == 1 or lex or ling or quant

    # Log disagreement if ML disagrees with rule-based detection
    if ml_pred != final:
        disagreement_log.append({
            "text": text,
            "type": "superlative",
            "ml_superlative": ml_pred,
            "lexical_superlative": lex,
            "linguistic_superlative": ling,
            "quantifier_superlative": quant,
            "final_superlative": final,
        })

    return final, ml_pred, lex, ling, quant


# ============================================================
# CONFORMANCE CHECKING (S‑B: line‑by‑line comments)
# ============================================================

def check_conformance(requirement: str, disagreement_log: list):
    """
    Performs full conformance checking:
        - Negation (ML)
        - Voice (hybrid ML + grammar)
        - Superlatives (hybrid ML + lexical + linguistic + quantifier)
        - Rule evaluation
    """

    # Load classifiers
    neg_clf, voice_clf, super_clf = load_classifiers()

    # Encode requirement for ML classifiers
    emb = model.encode([requirement])

    # NEGATION (ML only)
    is_negative = bool(neg_clf.predict(emb)[0])

    # VOICE (hybrid)
    final_passive, p_passive, voice_source = hybrid_voice_decision(
        requirement, voice_clf, disagreement_log
    )
    is_active_voice = not final_passive

    # SUPERLATIVES (hybrid)
    final_super, ml_super, lex_super, ling_super, quant_super = hybrid_superlative_decision(
        requirement, super_clf, disagreement_log
    )

    # Rule evaluations
    conforms_rule_1 = not is_negative
    conforms_rule_2 = is_active_voice
    conforms_rule_3 = not final_super

    # Overall conformance
    overall = conforms_rule_1 and conforms_rule_2 and conforms_rule_3

    return {
        "requirement": requirement,
        "is_negative": is_negative,
        "is_active_voice": is_active_voice,
        "has_superlative": final_super,
        "voice_decision_source": voice_source,
        "voice_ml_p_passive": p_passive,
        "ml_superlative": ml_super,
        "lexical_superlative": lex_super,
        "linguistic_superlative": ling_super,
        "quantifier_superlative": quant_super,
        "conforms_to_rule_1": conforms_rule_1,
        "conforms_to_rule_2": conforms_rule_2,
        "conforms_to_rule_3": conforms_rule_3,
        "overall_conformance": overall,
    }


# -----------------------------
# MAIN EXECUTION
# -----------------------------

if __name__ == "__main__":

    print("\nRunning k-fold evaluation...")
    run_kfold_evaluation(k=5)

    print("\nTraining final classifiers...")
    train_classifiers_csv()

    disagreements = []

    print("\nEnter requirements for conformance checking (hybrid ML + grammar + superlatives).")

    while True:
        req = input("\nRequirement (or 'exit'): ")
        if req.lower() == "exit":
            break

        result = check_conformance(req, disagreements)

        print("\n--- Conformance Result ---")
        for k, v in result.items():
            print(f"{k}: {v}")

    if disagreements:
        pd.DataFrame(disagreements).to_csv("disagreements.csv", index=False)
        print("\nLogged disagreements to disagreements.csv")
