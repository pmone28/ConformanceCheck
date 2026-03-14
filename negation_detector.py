from spacy_loader import nlp

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
