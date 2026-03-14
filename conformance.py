from sbert import model
from negation_detector import extract_negation_cues
from voice_detector import hybrid_voice_decision, extract_passive_indicators
from superlative_detector import hybrid_superlative_decision, extract_superlative_cues
from subjective_detector import subjective_decision
from vaguePronouns_detector import vaguePronouns_decision
from ambiguousAd_detector import ambiguousAd_decision
from compPhrases_detector import compPhrases_decision
from loopholes_detector import loophole_decision
from openEnded_detector import openEnd_decision

def check_conformance(requirement, disagreement_log, neg_clf, voice_clf, super_clf, subjective_clf, vagueP_clf, ambiguousAd_clf, compPhrase_clf, loophole_clf, openEnd_clf):

    emb = model.encode([requirement])

    is_negative = bool(neg_clf.predict(emb)[0])

    final_passive, p_passive, voice_source = hybrid_voice_decision(
        requirement, voice_clf, disagreement_log
    )
    is_active_voice = not final_passive

    final_super, ml_super, lex_super, ling_super, quant_super = hybrid_superlative_decision(requirement, super_clf, disagreement_log)

    is_subjective = bool(subjective_decision(requirement, subjective_clf))
    
    is_vagueP = bool(vaguePronouns_decision(requirement, vagueP_clf))
    
    is_ambiguousAd = bool(ambiguousAd_decision(requirement, ambiguousAd_clf))
    
    is_compPhrase = bool(compPhrases_decision(requirement, compPhrase_clf))
    
    is_loophole = bool(loophole_decision(requirement, loophole_clf))
    
    is_openEnd = bool(openEnd_decision(requirement, openEnd_clf))
    
    conforms_rule_1 = not is_negative
    conforms_rule_2 = is_active_voice
    conforms_rule_3 = not final_super
    conforms_rule_4 = not is_subjective
    conforms_rule_5 = not is_vagueP
    conforms_rule_6 = not is_ambiguousAd
    conforms_rule_7 = not is_compPhrase
    conforms_rule_8 = not is_loophole
    conforms_rule_9 = not is_openEnd

    overall = conforms_rule_1 and conforms_rule_2 and conforms_rule_3 and conforms_rule_4 and conforms_rule_5 and conforms_rule_6 and conforms_rule_7 and conforms_rule_8 and conforms_rule_9

    negation_cues = extract_negation_cues(requirement)
    passive_indicators = extract_passive_indicators(requirement)
    superlative_cues = extract_superlative_cues(requirement)

    return {
        "is_negative": is_negative,
        "is_active_voice": is_active_voice,
        "has_superlative": final_super,
        "has_subjective_language":is_subjective,
        "has_vague_pronouns":is_vagueP,
        "has_ambiguous_adverbs_adjectives":is_ambiguousAd,
        "has_comparative phrases":is_compPhrase,
        "has_loopholes":is_loophole,
        "has_openEnded_statements ":is_openEnd,
        "voice_decision_source": voice_source,
        "voice_ml_p_passive": p_passive,
        "ml_superlative": ml_super,
        "lexical_superlative": lex_super,
        "linguistic_superlative": ling_super,
        "quantifier_superlative": quant_super,
        "conforms_to_rule_1_Negative": conforms_rule_1,
        "conforms_to_rule_2_ActiveVoice": conforms_rule_2,
        "conforms_to_rule_3_Superlative": conforms_rule_3,
        "conforms_to_rule_4_Subjective": conforms_rule_4,
        "conforms_to_rule_5_VaguePronouns": conforms_rule_5,
        "conforms_to_rule_6_AmbiguousAd": conforms_rule_6,
        "conforms_to_rule_7_CompPhrase": conforms_rule_7,
        "conforms_to_rule_8_Loopholes": conforms_rule_8,
        "conforms_to_rule_9_OpenEnded": conforms_rule_9,
        "overall_conformance": overall,
        "negation_cues": negation_cues,
        "passive_indicators": passive_indicators,
        "superlative_cues": superlative_cues,
    }
