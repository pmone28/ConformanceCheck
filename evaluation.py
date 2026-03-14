import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sbert import model
from training import load_training_data_csv

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
    # Now loads 4 datasets
    neg_df, voice_df, super_df, subj_df, vagueP_df, ambiguousAd_df, compPhrase_df, loophole_df, openEnded_df = load_training_data_csv()

    # NEGATION
    X_neg = model.encode(neg_df["text"].astype(str).tolist())
    y_neg = neg_df["label"].astype(int).values
    print("\nNEGATION:", evaluate_classifier_kfold(X_neg, y_neg, k))

    # VOICE
    X_voice = model.encode(voice_df["text"].astype(str).tolist())
    y_voice = voice_df["label"].astype(int).values
    print("\nVOICE:", evaluate_classifier_kfold(X_voice, y_voice, k))

    # SUPERLATIVES
    X_super = model.encode(super_df["text"].astype(str).tolist())
    y_super = super_df["label"].astype(int).values
    print("\nSUPERLATIVES:", evaluate_classifier_kfold(X_super, y_super, k))

    # SUBJECTIVE LANGUAGE
    X_subj = model.encode(subj_df["text"].astype(str).tolist())
    y_subj = subj_df["label"].astype(int).values
    print("\nSUBJECTIVE LANGUAGE:", evaluate_classifier_kfold(X_subj, y_subj, k))
    
    # VAGUE PRONOUNS
    X_vagueP = model.encode(vagueP_df["text"].astype(str).tolist())
    y_vagueP = vagueP_df["label"].astype(int).values
    print("\nVAGUE PRONOUNS:", evaluate_classifier_kfold(X_vagueP, y_vagueP, k))
    
    # AMBIGUOUS ADVERBS AND ADJECTIVES 
    X_ambiguousAd = model.encode(ambiguousAd_df["text"].astype(str).tolist())
    y_ambiguousAd = ambiguousAd_df["label"].astype(int).values
    print("\nAMBIGUOUS ADVERBS AND ADJECTIVES:", evaluate_classifier_kfold(X_ambiguousAd, y_ambiguousAd, k))
    
    # COMPARATIVE PHRASES 
    X_compPhrase = model.encode(compPhrase_df["text"].astype(str).tolist())
    y_compPhrase = compPhrase_df["label"].astype(int).values
    print("\nCOMPARATIVE PHRASES:", evaluate_classifier_kfold(X_compPhrase, y_compPhrase, k))
    
    # LOOPHOLES 
    X_loophole = model.encode(loophole_df["text"].astype(str).tolist())
    y_loophole = loophole_df["label"].astype(int).values
    print("\nLOOPHOLES:", evaluate_classifier_kfold(X_loophole, y_loophole, k))
    
    # OPEN ENDED 
    X_openEnd = model.encode(openEnded_df["text"].astype(str).tolist())
    y_openEnd = openEnded_df["label"].astype(int).values
    print("\nOPEN ENDED:", evaluate_classifier_kfold(X_openEnd, y_openEnd, k))
