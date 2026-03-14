from sbert import model

def openEnd_decision(text, clf):
    emb = model.encode([text])
    pred = int(clf.predict(emb)[0])
    return pred  # 1 = open ended, 0 = non open ended
