import joblib

def load_classifiers():
    neg = joblib.load("negation_classifier.pkl")
    voice = joblib.load("voice_classifier.pkl")
    superl = joblib.load("superlatives_classifier.pkl")
    subLan = joblib.load("subjective_classifier.pkl")
    vagueP = joblib.load("vagueP_classifier.pkl")
    ambiguousAd = joblib.load("ambiguousAd_classifier.pkl")
    compPhrase = joblib.load("compPhrase_classifier.pkl")
    loophole = joblib.load("loophole_classifier.pkl")
    openEnd = joblib.load("openEnd_classifier.pkl")
    
    return neg, voice, superl,subLan, vagueP, ambiguousAd, compPhrase, loophole, openEnd
