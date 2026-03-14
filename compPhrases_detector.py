from sbert import model

def compPhrases_decision(text, clf):
    emb = model.encode([text])
    pred = int(clf.predict(emb)[0])
    return pred  # 1 = comparative, 0 = non comparative
