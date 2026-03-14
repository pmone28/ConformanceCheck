import os

def classifiers_exist():
    return (
        os.path.exists("negation_classifier.pkl") and
        os.path.exists("voice_classifier.pkl") and
        os.path.exists("superlatives_classifier.pkl") and
        os.path.exists("subjective_classifier.pkl") and
        os.path.exists("vagueP_classifier.pkl") and
        os.path.exists("ambiguousAd_classifier.pkl") and
        os.path.exists("compPhrase_classifier.pkl") and
        os.path.exists("loophole_classifier.pkl") and
        os.path.exists("openEnd_classifier.pkl")
    )
