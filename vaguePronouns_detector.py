from sbert import model

def vaguePronouns_decision(text, clf):
    emb = model.encode([text])
    pred = int(clf.predict(emb)[0])
    return pred  # 1 = vague, 0 = notVague
