#!/usr/bin/env python3
"""
evaluate_pipeline.py (hardcoded mode - updated for new project layout)

Usage:
    python evaluate_pipeline.py

This script expects CSV inputs to live in the `csv files/` directory (created by main.py).
It will read:
 - csv files/references.csv
 - csv files/pipeline_transcripts.csv
 - csv files/labels_gold.csv
 - csv files/segments_pred.csv
 - csv files/summary_refs.csv
 - csv files/summary_preds.csv

And it will write an evaluation JSON to:
 - csv files/evaluation_report.json

Dependencies:
    pip install pandas numpy scikit-learn jiwer rouge-score
"""
import os
import json
from typing import Dict

import numpy as np
import pandas as pd

from jiwer import wer
from sklearn.metrics import classification_report
from rouge_score import rouge_scorer

# HARDCODED PATHS 
CSV_DIR = "csv files"  # directory where main.py writes CSV outputs
# input files (inside CSV_DIR)
REFERENCES_CSV = os.path.join(CSV_DIR, "references.csv")
PIPELINE_TRANSCRIPTS_CSV = os.path.join(CSV_DIR, "pipeline_transcripts.csv")
LABELS_GOLD_CSV = os.path.join(CSV_DIR, "labels_gold.csv")
SEGMENTS_PRED_CSV = os.path.join(CSV_DIR, "segments_pred.csv")
SUMMARY_REFS_CSV = os.path.join(CSV_DIR, "summary_refs.csv")
SUMMARY_PREDS_CSV = os.path.join(CSV_DIR, "summary_preds.csv")
# output
OUTPUT_JSON = os.path.join(CSV_DIR, "evaluation_report.json")



def compute_wer_mean_median(refs_csv: str, hyps_csv: str, call_id_col: str = "call_id") -> Dict[str, float]:
    """Compute WER per call, return mean and median."""
    if not os.path.exists(refs_csv):
        raise FileNotFoundError(f"Missing references file: {refs_csv}")
    if not os.path.exists(hyps_csv):
        raise FileNotFoundError(f"Missing hypotheses file: {hyps_csv}")

    refs = pd.read_csv(refs_csv).set_index(call_id_col)
    hyps = pd.read_csv(hyps_csv).set_index(call_id_col)
    common = refs.index.intersection(hyps.index)

    wers = []
    for cid in common:
        r = str(refs.loc[cid, refs.columns[0]])  # first column assumed to be the reference text field
        h = str(hyps.loc[cid, hyps.columns[0]])  # first column assumed to be the hypothesis text field
        try:
            w = wer(r, h)
        except Exception:
            w = float("nan")
        wers.append(w)

    wers_arr = np.array([w for w in wers if not np.isnan(w)])
    if len(wers_arr) == 0:
        return {"count": 0, "mean_wer": None, "median_wer": None}
    return {"count": int(len(wers_arr)), "mean_wer": float(np.mean(wers_arr)), "median_wer": float(np.median(wers_arr))}


def compute_classification_metrics(gold_csv: str, pred_csv: str, index_col: str = "segment_id") -> Dict[str, Dict]:
    """
    Compute classification reports for sentiment and intent.
    """
    if not os.path.exists(gold_csv):
        raise FileNotFoundError(f"Missing gold labels file: {gold_csv}")
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"Missing prediction labels file: {pred_csv}")

    gold = pd.read_csv(gold_csv).set_index(index_col)
    pred = pd.read_csv(pred_csv).set_index(index_col)
    df = gold.join(pred, how="inner", rsuffix="_pred")

    results = {}
    # Sentiment
    if "sentiment" in df.columns and "sentiment_pred" in df.columns:
        report_sentiment = classification_report(df["sentiment"], df["sentiment_pred"], digits=3, zero_division=0)
        results["sentiment"] = {"rows": int(len(df)), "report": report_sentiment}
    else:
        results["sentiment"] = {"rows": int(len(df)), "report": "Missing columns for sentiment or sentiment_pred"}

    # Intent
    if "intent" in df.columns and "intent_pred" in df.columns:
        report_intent = classification_report(df["intent"], df["intent_pred"], digits=3, zero_division=0)
        results["intent"] = {"rows": int(len(df)), "report": report_intent}
    else:
        results["intent"] = {"rows": int(len(df)), "report": "Missing columns for intent or intent_pred"}

    return results


def compute_rouge_mean(refs_csv: str, preds_csv: str, id_col: str = "call_id", ref_col: str = "reference_summary", pred_col: str = "predicted_summary") -> Dict[str, float]:
    """
    Compute mean ROUGE-1 and ROUGE-L f-measure across matching call IDs.
    """
    if not os.path.exists(refs_csv) or not os.path.exists(preds_csv):
        return {"count": 0, "mean_rouge1": None, "mean_rougeL": None}

    refs = pd.read_csv(refs_csv).set_index(id_col)
    preds = pd.read_csv(preds_csv).set_index(id_col)
    common = refs.index.intersection(preds.index)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_scores = []
    rL_scores = []
    for cid in common:
        r = str(refs.loc[cid, ref_col])
        p = str(preds.loc[cid, pred_col])
        try:
            sc = scorer.score(r, p)
            r1_scores.append(sc["rouge1"].fmeasure)
            rL_scores.append(sc["rougeL"].fmeasure)
        except Exception:
            continue

    if len(r1_scores) == 0:
        return {"count": 0, "mean_rouge1": None, "mean_rougeL": None}
    return {"count": len(r1_scores), "mean_rouge1": float(np.mean(r1_scores)), "mean_rougeL": float(np.mean(rL_scores))}


def run_evaluation():
    report = {"inputs": {}, "metrics": {}}

    # WER
    try:
        report["inputs"]["references_csv"] = REFERENCES_CSV
        report["inputs"]["pipeline_transcripts_csv"] = PIPELINE_TRANSCRIPTS_CSV
        wer_res = compute_wer_mean_median(REFERENCES_CSV, PIPELINE_TRANSCRIPTS_CSV)
        report["metrics"]["wer"] = wer_res
    except FileNotFoundError as e:
        print(f"[WARN] WER skipped: {e}")
        report["metrics"]["wer"] = None

    # Classification
    try:
        report["inputs"]["labels_gold_csv"] = LABELS_GOLD_CSV
        report["inputs"]["segments_pred_csv"] = SEGMENTS_PRED_CSV
        class_res = compute_classification_metrics(LABELS_GOLD_CSV, SEGMENTS_PRED_CSV)
        report["metrics"]["classification"] = class_res
    except FileNotFoundError as e:
        print(f"[WARN] Classification eval skipped: {e}")
        report["metrics"]["classification"] = None

    # ROUGE
    try:
        report["inputs"]["summary_refs_csv"] = SUMMARY_REFS_CSV
        report["inputs"]["summary_preds_csv"] = SUMMARY_PREDS_CSV
        rouge_res = compute_rouge_mean(SUMMARY_REFS_CSV, SUMMARY_PREDS_CSV)
        report["metrics"]["rouge"] = rouge_res
    except Exception as e:
        print(f"[WARN] ROUGE eval skipped or error: {e}")
        report["metrics"]["rouge"] = None

    # Print a human-readable summary
    print("\n=== Evaluation Summary ===\n")
    if report["metrics"].get("wer"):
        w = report["metrics"]["wer"]
        if w["mean_wer"] is not None:
            print(f"WER evaluated on {w['count']} calls  |  mean WER: {w['mean_wer']:.3f}  median WER: {w['median_wer']:.3f}")
        else:
            print("WER: (no valid WER values computed)")
    else:
        print("WER: (not evaluated)")

    if report["metrics"].get("classification"):
        print("\n-- Sentiment & Intent classification reports --\n")
        sentiment_report = report["metrics"]["classification"].get("sentiment", {})
        intent_report = report["metrics"]["classification"].get("intent", {})
        print("Sentiment report (rows evaluated: {})\n".format(sentiment_report.get("rows", "NA")))
        print(sentiment_report.get("report", "No sentiment report available"))
        print("\nIntent report (rows evaluated: {})\n".format(intent_report.get("rows", "NA")))
        print(intent_report.get("report", "No intent report available"))
    else:
        print("\nClassification: (not evaluated)")

    if report["metrics"].get("rouge"):
        r = report["metrics"]["rouge"]
        if r["mean_rouge1"] is not None:
            print(f"\nROUGE evaluated on {r['count']} pairs  |  mean ROUGE-1: {r['mean_rouge1']:.3f}  mean ROUGE-L: {r['mean_rougeL']:.3f}")
        else:
            print("\nROUGE: (no valid ROUGE values computed)")
    else:
        print("\nROUGE: (not evaluated)")

    # Save a JSON report (placed inside csv files/ for convenient grouping)
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nSaved evaluation report to: {OUTPUT_JSON}")
    except Exception as e:
        print(f"[WARN] Failed to save report JSON: {e}")


if __name__ == "__main__":
    print("Running evaluation (reading CSVs from 'csv files/' directory). Edit the top of this script to change inputs.")
    run_evaluation()
