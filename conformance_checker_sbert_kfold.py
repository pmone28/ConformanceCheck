############################################################
# HYBRID REQUIREMENT CONFORMANCE CHECKER (TRAIN-ONCE VERSION)
############################################################

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

############################################################
# LOAD MODELS
############################################################

model = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")

############################################################
# TRAIN-ONCE CHECK
############################################################

def classifiers_exist():
    return (
        os.path.exists("negation_classifier.pkl") and
        os.path.exists("voice_classifier.pkl") and
        os.path.exists("superlatives_classifier.pkl")
    )

############################################################
# LOAD TRAINING DATA
############################################################

def load_training_data_csv(
    neg_csv="negation.csv",
    voice_csv="voice.csv",
    super_csv="superlatives.csv"
):
    for path in [neg_csv, voice_csv, super_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    neg_df = pd.read_csv(neg_csv, encoding="utf-8-sig")
    voice_df = pd.read_csv(voice_csv, encoding="utf-8-sig")
    super_df = pd.read_csv(super_csv, encoding="utf-8-sig")

    neg_df.columns = neg_df.columns.str.strip()
    voice_df.columns = voice_df.columns.str.strip()
    super_df.columns = super_df.columns.str.strip()

    return neg_df, voice_df, super_df

############################################################
# PASSIVE VOICE (GRAMMAR)
############################################################

def is_passive_grammar(text: str) -> bool:
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ("auxpass", "nsubjpass"):
            return True
    return False

############################################################
# K-FOLD EVALUATION
############################################################

def evaluate_classifier_kfold(embeddings, labels, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accuracies, precisions, recalls, f1s = [], [], [], []

    for train_idx, test_idx in kf.split(embeddings):
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=200, random_state=42)
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
    neg_df, voice_df, super_df = load_training_data_csv()

    X_neg = model.encode(neg_df["text"].astype(str).tolist())
    y_neg = neg_df["label"].astype(int).values
    print("\nNEGATION:", evaluate_classifier_kfold(X_neg, y_neg, k))

    X_voice = model.encode(voice_df["text"].astype(str).tolist())
    y_voice = voice_df["label"].astype(int).values
    print("\nVOICE:", evaluate_classifier_kfold(X_voice, y_voice, k))

    X_super = model.encode(super_df["text"].astype(str).tolist())
    y_super = super_df["label"].astype(int).values
    print("\nSUPERLATIVES:", evaluate_classifier_kfold(X_super, y_super, k))

############################################################
# TRAIN FINAL CLASSIFIERS
############################################################

def train_classifiers_csv():
    neg_df, voice_df, super_df = load_training_data_csv()

    # NEGATION
    X_neg = model.encode(neg_df["text"].astype(str).tolist())
    y_neg = neg_df["label"].astype(int).values
    neg_clf = LogisticRegression(max_iter=200, random_state=42)
    neg_clf.fit(X_neg, y_neg)
    joblib.dump(neg_clf, "negation_classifier.pkl")

    # VOICE
    X_voice = model.encode(voice_df["text"].astype(str).tolist())
    y_voice = voice_df["label"].astype(int).values
    voice_clf = LogisticRegression(max_iter=200, random_state=42)
    voice_clf.fit(X_voice, y_voice)
    joblib.dump(voice_clf, "voice_classifier.pkl")

    # SUPERLATIVES
    X_super = model.encode(super_df["text"].astype(str).tolist())
    y_super = super_df["label"].astype(int).values
    super_clf = LogisticRegression(max_iter=200, random_state=42)
    super_clf.fit(X_super, y_super)
    joblib.dump(super_clf, "superlatives_classifier.pkl")

    print("\nClassifiers trained and saved.")

def load_classifiers():
    return (
        joblib.load("negation_classifier.pkl"),
        joblib.load("voice_classifier.pkl"),
        joblib.load("superlatives_classifier.pkl"),
    )

############################################################
# HYBRID VOICE DECISION
############################################################

def hybrid_voice_decision(text: str, voice_clf, disagreement_log: list, conf_threshold=0.6):
    emb = model.encode([text])
    proba = voice_clf.predict_proba(emb)[0]
    p_passive = proba[1]
    ml_passive = p_passive >= 0.5
    grammar_passive = is_passive_grammar(text)

    if p_passive >= conf_threshold or p_passive <= (1 - conf_threshold):
        final_passive = ml_passive
        source = "ML"
    else:
        final_passive = grammar_passive
        source = "GRAMMAR"

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

############################################################
# SUPERLATIVE DETECTORS
############################################################

LEXICAL_SUPERLATIVES = {
    "best", "worst", "maximum", "minimum", "optimal",
    "topmost", "foremost", "paramount", "unmatched",
    "unparalleled", "superior", "supreme"
}

def lexical_superlative_detector(text: str) -> bool:
    return any(tok in LEXICAL_SUPERLATIVES for tok in text.lower().split())

def linguistic_superlative_detector(text: str) -> bool:
    doc = nlp(text)
    for i, token in enumerate(doc):
        if token.tag_ in ("JJS", "RBS"):
            return True
        if token.lower_ == "most" and i + 1 < len(doc) and doc[i+1].pos_ == "ADJ":
            return True
        if token.lower_ == "least" and i + 1 < len(doc) and doc[i+1].pos_ == "ADJ":
            return True
    return False

def quantifier_superlative_detector(text: str) -> bool:
    doc = nlp(text)
    for i, token in enumerate(doc):
        if re.match(r"^(most|least)-", token.text.lower()):
            return True
        if token.lower_ in ("most", "least"):
            if i + 1 < len(doc) and doc[i+1].lower_ == "of":
                return True
            if i + 1 < len(doc) and doc[i+1].pos_ in ("NOUN", "PROPN"):
                return True
    return False

# ---------------------------------------------------------
# HYBRID SUPERLATIVE DECISION
# ---------------------------------------------------------

def hybrid_superlative_decision(text: str, super_clf, disagreement_log: list):
    emb = model.encode([text])
    ml_pred = int(super_clf.predict(emb)[0])

    # Rule-based detectors
    lex = lexical_superlative_detector(text)
    ling = linguistic_superlative_detector(text)
    quant = quantifier_superlative_detector(text)

    # Hybrid final decision
    final = ml_pred == 1 or lex or ling or quant

    # Log disagreements
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

############################################################
# EXPLANATION CUES
############################################################

def extract_negation_cues(text: str):
    doc = nlp(text)
    cues = []

    for token in doc:
        if token.dep_ == "neg":
            cues.append(token.text)

    lexical_neg = ["no", "not", "never", "without", "cannot", "can't", "won't", "shall not", "must not"]
    for word in lexical_neg:
        if word in text.lower():
            cues.append(word)

    return list(set(cues))

def extract_passive_indicators(text: str):
    doc = nlp(text)
    indicators = []

    for token in doc:
        if token.dep_ == "auxpass":
            indicators.append(f"auxpass:{token.text}")
        if token.dep_ == "nsubjpass":
            indicators.append(f"nsubjpass:{token.text}")
        if token.tag_ == "VBN" and token.head.dep_ == "auxpass":
            indicators.append(f"participle:{token.text}")

    return indicators

def extract_superlative_cues(text: str):
    doc = nlp(text)
    cues = []

    for word in LEXICAL_SUPERLATIVES:
        if word in text.lower():
            cues.append(f"lexical:{word}")

    for token in doc:
        if token.tag_ in ("JJS", "RBS"):
            cues.append(f"linguistic:{token.text}")

    for i, token in enumerate(doc):
        if token.lower_ in ("most", "least"):
            cues.append(f"quantifier:{token.text}")
            if i + 1 < len(doc):
                cues.append(f"context:{token.text} {doc[i+1].text}")

    return cues

############################################################
# CONFORMANCE CHECK
############################################################

def check_conformance(requirement: str,
                      disagreement_log: list,
                      neg_clf,
                      voice_clf,
                      super_clf):

    emb = model.encode([requirement])

    is_negative = bool(neg_clf.predict(emb)[0])

    final_passive, p_passive, voice_source = hybrid_voice_decision(
        requirement, voice_clf, disagreement_log
    )
    is_active_voice = not final_passive

    final_super, ml_super, lex_super, ling_super, quant_super = hybrid_superlative_decision(
        requirement, super_clf, disagreement_log
    )

    conforms_rule_1 = not is_negative
    conforms_rule_2 = is_active_voice
    conforms_rule_3 = not final_super

    overall = conforms_rule_1 and conforms_rule_2 and conforms_rule_3

    # Explanation cues
    negation_cues = extract_negation_cues(requirement)
    passive_indicators = extract_passive_indicators(requirement)
    superlative_cues = extract_superlative_cues(requirement)

    return {
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
        "negation_cues": negation_cues,
        "passive_indicators": passive_indicators,
        "superlative_cues": superlative_cues,
    }

############################################################
# BATCH PROCESSING
############################################################

def batch_process_csv(
    input_csv="ESA-NATH-WFI-RS-001_v1_extracted.csv",
    output_csv="ESA-NATH-WFI-RS-001_v1_conformance.csv"
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if "id" not in df.columns or "raw_text" not in df.columns:
        raise ValueError("Input CSV must contain 'id' and 'raw_text' columns.")

    neg_clf, voice_clf, super_clf = load_classifiers()
    disagreements = []
    output_rows = []

    for _, row in df.iterrows():
        req_id = row["id"]
        text = row["raw_text"]

        if not isinstance(text, str) or not text.strip():
            continue

        text = " ".join(text.split())

        result = check_conformance(
            text,
            disagreements,
            neg_clf,
            voice_clf,
            super_clf
        )

        output_rows.append({
            "id": req_id,
            "raw_text": text,
            **result
        })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nBatch results written to: {output_csv}")

    if disagreements:
        pd.DataFrame(disagreements).to_csv("disagreements.csv", index=False, encoding="utf-8-sig")
        print("Disagreements logged to disagreements.csv")

############################################################
# MAIN (TRAIN ONCE, THEN REUSE)
############################################################

if __name__ == "__main__":
    print("\n=== Hybrid Conformance Checker ===")

    if not classifiers_exist():
        print("\nNo trained classifiers found. Training models for the first time...")
        run_kfold_evaluation(k=5)
        train_classifiers_csv()
        print("Training complete.")
    else:
        print("\nClassifiers already exist. Skipping training.")

    print("\nRunning batch conformance checking...")
    batch_process_csv()

    print("\nDone.")
