import re
from spacy_loader import nlp
from sbert import model

LEXICAL_SUPERLATIVES = {
    "best", "worst", "maximum", "minimum", "optimal",
    "topmost", "foremost", "paramount", "unmatched",
    "unparalleled", "superior", "supreme"
}

def lexical_superlative_detector(text: str):
    return any(tok in LEXICAL_SUPERLATIVES for tok in text.lower().split())


def linguistic_superlative_detector(text: str):
    doc = nlp(text)
    for i, token in enumerate(doc):
        if token.tag_ in ("JJS", "RBS"):
            return True
        if token.lower_ == "most" and i + 1 < len(doc) and doc[i+1].pos_ == "ADJ":
            return True
        if token.lower_ == "least" and i + 1 < len(doc) and doc[i+1].pos_ == "ADJ":
            return True
    return False


def quantifier_superlative_detector(text: str):
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


def hybrid_superlative_decision(text, super_clf, disagreement_log):
    emb = model.encode([text])
    ml_pred = int(super_clf.predict(emb)[0])

    lex = lexical_superlative_detector(text)
    ling = linguistic_superlative_detector(text)
    quant = quantifier_superlative_detector(text)

    final = ml_pred == 1 or lex or ling or quant

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
