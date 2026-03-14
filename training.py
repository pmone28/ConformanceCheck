import os
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sbert import model

def load_training_data_csv(
    neg_csv="training_negation.csv",
    voice_csv="training_voice.csv",
    super_csv="training_superlatives.csv",
    subj_csv="training_subLang.csv",
    vaguePronouns_csv="training_vaguePronouns.csv",
    ambiguousAd_csv="training_AmbiguousAd.csv",
    compPhrase_csv="training_ComparativePhrases.csv",
    loophole_csv="training_Loopholes.csv",
    OpenEnded_csv="training_OpenEnded.csv"
):
    # Check all required files exist
    for path in [neg_csv, voice_csv, super_csv, subj_csv, vaguePronouns_csv, ambiguousAd_csv, compPhrase_csv, loophole_csv, OpenEnded_csv]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    # Load datasets
    neg_df = pd.read_csv(neg_csv, encoding="utf-8-sig")
    voice_df = pd.read_csv(voice_csv, encoding="utf-8-sig")
    super_df = pd.read_csv(super_csv, encoding="utf-8-sig")
    subj_df = pd.read_csv(subj_csv, encoding="utf-8-sig")
    vagueP_df = pd.read_csv(vaguePronouns_csv, encoding="utf-8-sig")
    ambiguousAd_df = pd.read_csv(ambiguousAd_csv, encoding="utf-8-sig")
    compPhrase_df = pd.read_csv(compPhrase_csv, encoding="utf-8-sig")
    loophole_df = pd.read_csv(loophole_csv, encoding="utf-8-sig")
    openEnded_df = pd.read_csv(OpenEnded_csv, encoding="utf-8-sig")

    # Clean column names
    neg_df.columns = neg_df.columns.str.strip()
    voice_df.columns = voice_df.columns.str.strip()
    super_df.columns = super_df.columns.str.strip()
    subj_df.columns = subj_df.columns.str.strip()
    vagueP_df.columns = vagueP_df.columns.str.strip()
    ambiguousAd_df.columns = ambiguousAd_df.columns.str.strip()
    compPhrase_df.columns = compPhrase_df.columns.str.strip()
    loophole_df.columns = loophole_df.columns.str.strip()
    openEnded_df.columns = openEnded_df.columns.str.strip()
    
    return neg_df, voice_df, super_df, subj_df, vagueP_df, ambiguousAd_df, compPhrase_df, loophole_df, openEnded_df


def train_classifiers_csv():
    neg_df, voice_df, super_df, subj_df, vagueP_df, ambiguousAd_df, compPhrase_df, loophole_df, openEnded_df = load_training_data_csv()

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

    # SUBJECTIVE LANGUAGE
    X_subj = model.encode(subj_df["text"].astype(str).tolist())
    y_subj = subj_df["label"].astype(int).values
    subj_clf = LogisticRegression(max_iter=200, random_state=42)
    subj_clf.fit(X_subj, y_subj)
    joblib.dump(subj_clf, "subjective_classifier.pkl")
    
    # VAGUE PRONOUNS
    X_vagueP = model.encode(vagueP_df["text"].astype(str).tolist())
    y_vagueP = vagueP_df["label"].astype(int).values
    vagueP_clf = LogisticRegression(max_iter=200, random_state=42)
    vagueP_clf.fit(X_vagueP, y_vagueP)
    joblib.dump(vagueP_clf, "vagueP_classifier.pkl")

    # AMBIGUOUS ADVERBS AND ADJECTIVES 
    X_ambiguousAd = model.encode(ambiguousAd_df["text"].astype(str).tolist())
    y_ambiguousAd = ambiguousAd_df["label"].astype(int).values
    ambiguousAd_clf = LogisticRegression(max_iter=200, random_state=42)
    ambiguousAd_clf.fit(X_ambiguousAd, y_ambiguousAd)
    joblib.dump(ambiguousAd_clf, "ambiguousAd_classifier.pkl")
    
    # COMPARATIVE PHRASES 
    X_compPhrase = model.encode(compPhrase_df["text"].astype(str).tolist())
    y_compPhrase = compPhrase_df["label"].astype(int).values
    compPhrase_clf = LogisticRegression(max_iter=200, random_state=42)
    compPhrase_clf.fit(X_compPhrase, y_compPhrase)
    joblib.dump(compPhrase_clf, "compPhrase_classifier.pkl")
    
    # LOOPHOLES
    X_loophole = model.encode(loophole_df["text"].astype(str).tolist())
    y_loophole = loophole_df["label"].astype(int).values
    loophole_clf = LogisticRegression(max_iter=200, random_state=42)
    loophole_clf.fit(X_loophole, y_loophole)
    joblib.dump(loophole_clf, "loophole_classifier.pkl")
    
    # OPEN ENDED 
    X_openEnd = model.encode(openEnded_df["text"].astype(str).tolist())
    y_openEnd = openEnded_df["label"].astype(int).values
    openEnd_clf = LogisticRegression(max_iter=200, random_state=42)
    openEnd_clf.fit(X_openEnd, y_openEnd)
    joblib.dump(openEnd_clf, "openEnd_classifier.pkl")
    
    print("\nClassifiers trained and saved.")
