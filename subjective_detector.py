from sbert import model

def subjective_decision(text, clf):
    emb = model.encode([text])
    pred = int(clf.predict(emb)[0])
    return pred  # 1 = subjective, 0 = objective
