import os
import pandas as pd
from loading import load_classifiers
from conformance import check_conformance
from datetime import datetime


def export_rule_conformance_csv(results_df):
    # Uses the fields returned by your existing conformance.py:
    #   conforms_to_rule_1..4, overall_conformance

    export_data = {
        "id": results_df["id"],
        "raw_text": results_df["raw_text"],
        "Negative": results_df["conforms_to_rule_1_Negative"].astype(int),
        "Active Voice": results_df["conforms_to_rule_2_ActiveVoice"].astype(int),
        "Superlative": results_df["conforms_to_rule_3_Superlative"].astype(int),
        "Subjective Language": results_df["conforms_to_rule_4_Subjective"].astype(int),
        "Vague Pronouns": results_df["conforms_to_rule_5_VaguePronouns"].astype(int),
        "Ambiguous Adverbs and adjectives": results_df["conforms_to_rule_6_AmbiguousAd"].astype(int),
        "Comparative Phrases": results_df["conforms_to_rule_7_CompPhrase"].astype(int),
        "Loopholes": results_df["conforms_to_rule_8_Loopholes"].astype(int),
        "Open Ended Statements": results_df["conforms_to_rule_9_OpenEnded"].astype(int),
        "Overall Conformace": results_df["overall_conformance"].astype(int),
    }

    rule_df = pd.DataFrame(export_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #filename = f"Rule_Conformance_{timestamp}.csv"
    filename = f"Rule_Conformance.csv"

    rule_df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Rule-only conformance saved to: {filename}")


def batch_process_csv(
    input_csv="Input_Restructured.csv",
    output_csv=None
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if output_csv is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #output_csv = f"Rule_Conformance_Details_{timestamp}.csv"
        output_csv = f"Rule_Conformance_Details.csv"

    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    if "id" not in df.columns or "raw_text" not in df.columns:
        raise ValueError("Input CSV must contain 'id' and 'raw_text' columns.")

    neg_clf, voice_clf, super_clf, subjective_clf, vagueP_clf, ambiguousAd_clf, compPhrase_clf, loophole_clf, openEnd_clf = load_classifiers()
    disagreements = []
    output_rows = []

    for _, row in df.iterrows():
        req_id = row["id"]
        text = row["raw_text"]

        if not isinstance(text, str) or not text.strip():
            continue

        text = " ".join(text.split())

        result = check_conformance(
            text,
            disagreements,
            neg_clf,
            voice_clf,
            super_clf,
            subjective_clf,
            vagueP_clf,
            ambiguousAd_clf,
            compPhrase_clf,
            loophole_clf,
            openEnd_clf
        )

        output_rows.append({
            "id": req_id,
            "raw_text": text,
            **result
        })

    out_df = pd.DataFrame(output_rows)

    # 1) Full detailed CSV
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nFull batch results written to: {output_csv}")

    # 2) Conformance-only CSV
    export_rule_conformance_csv(out_df)

    if disagreements:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        disagreements_file = f"disagreements_{timestamp}.csv"
        pd.DataFrame(disagreements).to_csv(disagreements_file, index=False, encoding="utf-8-sig")
        print(f"Disagreements logged to {disagreements_file}")
