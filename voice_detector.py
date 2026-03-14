from spacy_loader import nlp
from sbert import model

def is_passive_grammar(text: str) -> bool:
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ("auxpass", "nsubjpass"):
            return True
    return False


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


def hybrid_voice_decision(text, voice_clf, disagreement_log, conf_threshold=0.6):
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
