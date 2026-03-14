from sbert import model

def loophole_decision(text, clf):
    emb = model.encode([text])
    pred = int(clf.predict(emb)[0])
    return pred  # 1 = has loopholes, 0 = no loopholes
