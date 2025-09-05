import os
import logging
from dotenv import load_dotenv
import warnings
import json

import numpy as np
import pandas as pd
from scipy.special import softmax

import whisperx
import torchaudio
from pydub import AudioSegment

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereEmbeddings

import requests

warnings.filterwarnings("ignore")


# FILE / FOLDER CONFIGURATION
CSV_DIR = "csv files"               # directory for CSV inputs/outputs
TEXT_OUT_DIR = "Text Output files"  # directory for text outputs (summary/next_steps)
# ensure directories exist
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(TEXT_OUT_DIR, exist_ok=True)

# Configuration / constants
DEFAULT_INTENT_LABELS = [
    "Greeting / Opening",
    "Self Introduction",
    "Polite Courtesy",
    "Asking Information / Query",
    "Providing Information",
    "Clarification / Doubt",
    "Requesting Service / Action",
    "Order / Purchase",
    "Cancellation Request",
    "Billing / Payment Issue",
    "Technical Support Request",
    "Escalation Request",
    "Complaint / Issue Report",
    "Feedback / Suggestion",
    "Closing / Goodbye",
    "Other / Irrelevant",
]

# Utilities
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_mono_chunk(audio_path, start_time, end_time):
    """Load a segment of audio as mono waveform between start_time and end_time (seconds).
    Returns: numpy array (1d) and sample rate.
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    chunk = waveform[:, start_sample:end_sample]
    return chunk.squeeze(0).numpy(), sr


# Pipeline
class CallAnalysisPipeline:
    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 16,
        compute_type: str = "int8",
        intent_labels: list = None,
    ):
        load_dotenv()
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type

        # envs
        self.HF_TOKEN = os.getenv("HF_TOKEN")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")

        # models placeholders
        self.transcribe_model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.sentiment_tokenizer = None
        self.sentiment_config = None
        self.sentiment_model = None
        self.audio_emotion = None
        self.intent_classifier = None

        # labels
        self.intent_labels = intent_labels or DEFAULT_INTENT_LABELS

        # Initialize heavy models lazily when needed
        logger.info("CallAnalysisPipeline initialized")

    # Model loading helpers
    def load_transcription(self, model_name: str = "medium"):
        if self.transcribe_model is None:
            logger.info("Loading WhisperX transcription model level: %s", model_name)
            self.transcribe_model = whisperx.load_model(
                model_name, self.device, compute_type=self.compute_type
            )
        return self.transcribe_model

    def load_alignment_model(self, language_code="en"):
        if self.align_model is None:
            logger.info("Loading WhisperX alignment model for language: %s", language_code)
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
        return self.align_model, self.align_metadata

    def load_diarization(self):
        if self.diarize_model is None:
            logger.info("Loading WhisperX diarization pipeline")
            self.diarize_model = whisperx.diarize.DiarizationPipeline(
                use_auth_token=self.HF_TOKEN, device=self.device
            )
        return self.diarize_model

    def load_sentiment_model(self, sentiment_model_id: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        if self.sentiment_model is None:
            logger.info("Loading sentiment model: %s", sentiment_model_id)
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_id)
            self.sentiment_config = AutoConfig.from_pretrained(sentiment_model_id)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_id)
        return self.sentiment_tokenizer, self.sentiment_config, self.sentiment_model

    def load_audio_emotion(self, model_id: str = "kushalballari/distilhubert-tone-classification"):
        if self.audio_emotion is None:
            logger.info("Loading audio emotion model: %s", model_id)
            # device=-1 uses CPU; keep consistent with caller
            self.audio_emotion = hf_pipeline(task="audio-classification", model=model_id, device=-1)
        return self.audio_emotion

    def load_intent_classifier(self, model_id: str = "joeddav/xlm-roberta-large-xnli"):
        if self.intent_classifier is None:
            logger.info("Loading zero-shot intent classifier: %s", model_id)
            self.intent_classifier = hf_pipeline("zero-shot-classification", model=model_id)
        return self.intent_classifier

    # Pipeline steps
    def transcribe(self, audio_file: str, language: str = None):
        """Transcribe audio file with WhisperX and return raw segments + full language detected."""
        model = self.load_transcription()
        logger.info("Loading audio: %s", audio_file)
        loaded_audio = whisperx.load_audio(audio_file)
        result = model.transcribe(loaded_audio, batch_size=self.batch_size, language=language or "en")
        segments = result.get("segments", [])
        language_detected = result.get("language", language or "en")
        logger.info("Transcription completed (%d segments).", len(segments))
        return segments, loaded_audio, language_detected

    def align_words(self, segments, loaded_audio, language_code="en"):
        """Run whisperx alignment to get word-level improved timestamps."""
        align_model, metadata = self.load_alignment_model(language_code=language_code)
        try:
            logger.info("Running alignment to improve word-level accuracy")
            aligned = whisperx.align(segments, align_model, metadata, loaded_audio, self.device, return_char_alignments=False)
        except Exception as e:
            logger.exception("Alignment failed: %s", e)
            aligned = {"segments": segments}
        return aligned

    def diarize_and_assign(self, loaded_audio, aligned_result):
        diarize_model = self.load_diarization()
        try:
            logger.info("Running diarization")
            diarize_segments = diarize_model(loaded_audio)
            assigned = whisperx.assign_word_speakers(diarize_segments, aligned_result)
            logger.info("Speaker assignment completed")
        except Exception as e:
            logger.exception("Diarization failed: %s", e)
            assigned = aligned_result
        return assigned

    def to_dataframe(self, segments_result):
        segments = segments_result.get("segments", [])
        df = pd.DataFrame(segments)
        # Keep only the columns that to be used downstream (start,end,text,speaker)
        for col in ["start", "end", "text", "speaker"]:
            if col not in df.columns:
                df[col] = None
        df = df[["start", "end", "text", "speaker"]]
        return df

    def analyze_sentiment(self, df: pd.DataFrame):
        tokenizer, config, model = self.load_sentiment_model()
        sentiments = []
        logger.info("Starting sentiment analysis on %d rows", len(df))
        for text in df["text"]:
            try:
                encoded_input = tokenizer(str(text), return_tensors="pt")
                output = model(**encoded_input)
                scores = output.logits[0].detach().numpy()
                scores = softmax(scores)
                top_idx = int(np.argmax(scores))
                label = config.id2label[top_idx]
                confidence = float(scores[top_idx])
                sentiments.append({"sentiment": label, "confidence": confidence})
            except Exception as e:
                logger.exception("Sentiment analysis failed for text: %s", str(text)[:50])
                sentiments.append({"sentiment": "Error", "confidence": 0.0})
        df["sentiment"] = [s["sentiment"] for s in sentiments]
        df["sentiment_conf"] = [s["confidence"] for s in sentiments]
        return df

    def analyze_tonality(self, df: pd.DataFrame, audio_file: str):
        audio_emotion = self.load_audio_emotion()
        tonalities = []
        logger.info("Starting tonality analysis on %d rows", len(df))
        for _, row in df.iterrows():
            try:
                chunk, sr = load_mono_chunk(audio_file, row["start"], row["end"])
                result = audio_emotion({"array": chunk, "sampling_rate": sr})
                if result:
                    top = max(result, key=lambda x: x["score"])
                    tonalities.append({"tonality": top["label"], "tonality_conf": float(top["score"])})
                else:
                    tonalities.append({"tonality": "Unknown", "tonality_conf": 0.0})
            except Exception as e:
                logger.exception("Tonality failed for row starting at %s: %s", row.get("start"), e)
                tonalities.append({"tonality": "Error", "tonality_conf": 0.0})
        df["tonality"] = [t["tonality"] for t in tonalities]
        df["tonality_conf"] = [t["tonality_conf"] for t in tonalities]
        return df

    def detect_intents(self, df: pd.DataFrame):
        classifier = self.load_intent_classifier()
        row_intents = []
        logger.info("Starting intent detection on %d rows", len(df))
        for text in df["text"]:
            try:
                res = classifier(str(text), self.intent_labels)
                row_intents.append({"intent": res["labels"][0], "intent_conf": float(res["scores"][0])})
            except Exception as e:
                logger.exception("Intent detection failed for text: %s", str(text)[:50])
                row_intents.append({"intent": "Error", "intent_conf": 0.0})
        df["intent"] = [ri["intent"] for ri in row_intents]
        df["intent_conf"] = [ri["intent_conf"] for ri in row_intents]
        return df

    def summarize_conversation(self, df: pd.DataFrame):
        full_conversation_text = "\n".join(df["text"].astype(str))
        if not self.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not configured; summarization skipped")
            return ""

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.OPENAI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an expert call center conversation summarizer.\n"
                        "Your job is to carefully analyze customer–agent dialogues and produce a structured summary.\n"
                        "Make sure the summary includes:\n- The main purpose of the call (complaint, query, request, etc.).\n- Key actions or issues discussed.\n- The tone and sentiment of the customer.\n- The resolution or next steps suggested by the agent.\n"
                        "Write the summary in clear, professional language, 3–5 sentences long, avoiding unnecessary details."
                    ),
                },
                {"role": "system", "content": f"Context:\n{full_conversation_text}"},
                {"role": "user", "content": "Summarize the text given"},
            ],
        }

        try:
            logger.info("Calling LLM for summarization")
            resp = requests.post(url, headers=headers, json=body, timeout=80)
            resp.raise_for_status()
            summarized_content = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception("LLM call failed for summarization: %s", e)
            summarized_content = ""
        return summarized_content

    def build_rag_index(self, docs_path: str = "docs.txt"):
        docs_path = docs_path if os.path.isabs(docs_path) else os.path.join(TEXT_OUT_DIR, docs_path) if os.path.exists(os.path.join(TEXT_OUT_DIR, docs_path)) else docs_path
        if not os.path.exists(docs_path):
            logger.warning("Policy docs not found at %s; skipping RAG index build", docs_path)
            return None
        if not self.COHERE_API_KEY:
            logger.warning("COHERE_API_KEY not configured; cannot build embeddings")
            return None

        logger.info("Building RAG index from %s", docs_path)
        loader = TextLoader(docs_path, encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=self.COHERE_API_KEY)
        vectorstore = FAISS.from_documents(docs, embeddings)
        self.vectorstore = vectorstore
        return vectorstore

    def retrieve_context(self, query: str, top_k: int = 3):
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            logger.warning("Vectorstore not available; returning empty policy context")
            return ""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(query)
        return "\n".join([r.page_content for r in results])

    def suggest_next_steps(self, summarized_content: str, policy_context: str):
        if not self.OPENAI_API_KEY:
            logger.warning("OPENAI_API_KEY not set; skipping next-steps suggestion")
            return ""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.OPENAI_API_KEY}", "Content-Type": "application/json"}
        body_next_steps = {
            "model": "mistralai/mistral-7b-instruct",
            "messages": [
                {"role": "system", "content": "You are an expert call center assistant.\nYour job is to recommend the next steps the agent should take, based ONLY on the conversation summary and provided company policies.\nKeep your recommendations clear, actionable, and professional."},
                {"role": "system", "content": f"Conversation Summary:\n{summarized_content}"},
                {"role": "system", "content": f"Policy Context:\n{policy_context}"},
                {"role": "user", "content": "Suggest follow-up actions for the agent based on the conversation summary and company policies."},
            ],
        }
        try:
            logger.info("Calling LLM to suggest next steps")
            resp_next = requests.post(url, headers=headers, json=body_next_steps, timeout=80)
            resp_next.raise_for_status()
            next_steps = resp_next.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception("LLM call failed (next steps): %s", e)
            next_steps = ""
        return next_steps

    # Orchestration
    def run_on_file(self, audio_file: str, language: str = None, docs_path: str = "docs.txt"):
        logger.info("Current file in Process: %s", audio_file)
        segments, loaded_audio, language_detected = self.transcribe(audio_file, language=language)
        aligned_result = self.align_words(segments, loaded_audio, language_code=language_detected)
        assigned = self.diarize_and_assign(loaded_audio, aligned_result)
        df = self.to_dataframe(assigned)

        # run analyses
        df = self.analyze_sentiment(df)
        df = self.analyze_tonality(df, audio_file=audio_file)
        df = self.detect_intents(df)

        logger.info("Analyses complete. %d rows in dataframe", len(df))

        # Summarize
        summary = self.summarize_conversation(df)
        logger.info("Summary: %s", (summary[:200] + "...") if summary else "(empty)")

        # Build RAG index and suggest next steps
        self.build_rag_index(docs_path=docs_path)
        policy_context = self.retrieve_context(summary, top_k=3)
        next_steps = self.suggest_next_steps(summary, policy_context)

        # Final results
        results = {
            "dataframe": df,
            "summary": summary,
            "policy_context": policy_context,
            "next_steps": next_steps,
        }
        return results


# start
if __name__ == "__main__":
    # Hardcoded configuration
    # place the audio file in project root (or give relative/absolute path)
    AUDIO_FILE = "ssvid.net---Hindi-Attending-Phone-Calls-Complaint-Call.mp3"
    DOCS_PATH = "docs.txt"
    DEVICE = "cpu"
    LANGUAGE = None
    BATCH_SIZE = 16
    COMPUTE_TYPE = "int8"

    # Initialize pipeline with the chosen settings
    pipeline = CallAnalysisPipeline(device=DEVICE, batch_size=BATCH_SIZE, compute_type=COMPUTE_TYPE)

    try:
        out = pipeline.run_on_file(AUDIO_FILE, language=LANGUAGE, docs_path=DOCS_PATH)

        # Export helper for evaluation (writes files expected by evaluate_pipeline.py)
        df_out = out["dataframe"].copy()

        # Ensure there is a segment_id column so downstream eval can join easily
        if "segment_id" not in df_out.columns:
            df_out.insert(0, "segment_id", [f"s{i+1}" for i in range(len(df_out))])

        # Decide which call_id to use:
        call_id = None
        try:
            refs_path = os.path.join(CSV_DIR, "references.csv")
            srefs_path = os.path.join(CSV_DIR, "summary_refs.csv")
            if os.path.exists(refs_path):
                refs_df = pd.read_csv(refs_path)
                if "call_id" in refs_df.columns and len(refs_df) > 0:
                    call_id = str(refs_df["call_id"].iloc[0])
            if call_id is None and os.path.exists(srefs_path):
                srefs_df = pd.read_csv(srefs_path)
                if "call_id" in srefs_df.columns and len(srefs_df) > 0:
                    call_id = str(srefs_df["call_id"].iloc[0])
        except Exception:
            call_id = None

        if call_id is None:
            call_id = os.path.splitext(os.path.basename(AUDIO_FILE))[0]

        # Save the per-segment CSV (pipeline predictions)
        segments_pred = df_out.copy()

        # Map pipeline column names to expected *_pred names
        rename_map = {}
        if "sentiment" in segments_pred.columns and "sentiment_pred" not in segments_pred.columns:
            rename_map["sentiment"] = "sentiment_pred"
        if "intent" in segments_pred.columns and "intent_pred" not in segments_pred.columns:
            rename_map["intent"] = "intent_pred"
        if "tonality" in segments_pred.columns and "tonality_pred" not in segments_pred.columns:
            rename_map["tonality"] = "tonality_pred"
        if rename_map:
            segments_pred = segments_pred.rename(columns=rename_map)

        # Ensure essential columns exist and add call_id
        for col in ["call_id", "start", "end", "speaker", "text"]:
            if col not in segments_pred.columns:
                if col == "call_id":
                    segments_pred[col] = call_id
                else:
                    segments_pred[col] = pd.NA

        # Reorder columns to expected order for evaluation convenience
        expected_cols = ["segment_id", "call_id", "start", "end", "speaker", "text", "sentiment_pred", "intent_pred", "tonality_pred"]
        for c in expected_cols:
            if c not in segments_pred.columns:
                segments_pred[c] = pd.NA
        segments_pred = segments_pred[expected_cols]

        # Write segments_pred.csv into CSV_DIR
        segments_pred_path = os.path.join(CSV_DIR, "segments_pred.csv")
        segments_pred.to_csv(segments_pred_path, index=False)
        print(f"[INFO] Wrote segments_pred.csv -> {segments_pred_path}")

        # Save the full per-call transcript for WER:
        transcript = " ".join(df_out["text"].astype(str).tolist()).replace("\n", " ").strip()
        pipeline_transcripts_path = os.path.join(CSV_DIR, "pipeline_transcripts.csv")
        pd.DataFrame([{"call_id": call_id, "transcript": transcript}]).to_csv(pipeline_transcripts_path, index=False)
        print(f"[INFO] Wrote pipeline_transcripts.csv -> {pipeline_transcripts_path} (call_id='{call_id}')")

        # Save analysis_output.csv as before (keeps existing behavior)
        analysis_output_path = os.path.join(CSV_DIR, "analysis_output.csv")
        df_out.to_csv(analysis_output_path, index=False)
        print(f"[INFO] Wrote analysis_output.csv -> {analysis_output_path}")

        # Save predicted summary for ROUGE eval (if summary exists)
        predicted_summary = out.get("summary", "") or ""
        summary_preds_path = os.path.join(CSV_DIR, "summary_preds.csv")
        pd.DataFrame([{"call_id": call_id, "predicted_summary": predicted_summary}]).to_csv(summary_preds_path, index=False)
        print(f"[INFO] Wrote summary_preds.csv -> {summary_preds_path}")

        # If a gold labels file exists in CSV_DIR, try to create a combined file to avoid separate preds
        labels_gold_path = os.path.join(CSV_DIR, "labels_gold.csv")
        if os.path.exists(labels_gold_path):
            try:
                gold = pd.read_csv(labels_gold_path)
                if "segment_id" in gold.columns:
                    combined = gold.merge(segments_pred, on="segment_id", how="left", suffixes=("","_pred"))

                    if "sentiment_pred" not in combined.columns and "sentiment" in segments_pred.columns:
                        combined = combined.rename(columns={"sentiment": "sentiment_pred"})
                    if "intent_pred" not in combined.columns and "intent" in segments_pred.columns:
                        combined = combined.rename(columns={"intent": "intent_pred"})

                    labels_combined_path = os.path.join(CSV_DIR, "labels_combined.csv")
                    combined.to_csv(labels_combined_path, index=False)
                    print(f"[INFO] Created labels_combined.csv -> {labels_combined_path}")
                else:
                    print("[WARN] Found labels_gold.csv but no 'segment_id' column. Skipping combined file creation.")
            except Exception as e:
                print(f"[WARN] Could not create labels_combined.csv: {e}")

        # Save summary.txt and next_steps.txt into TEXT_OUT_DIR
        summary_txt_path = os.path.join(TEXT_OUT_DIR, "summary.txt")
        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(predicted_summary or "")
        print(f"[INFO] Wrote summary.txt -> {summary_txt_path}")

        next_steps_txt_path = os.path.join(TEXT_OUT_DIR, "next_steps.txt")
        with open(next_steps_txt_path, "w", encoding="utf-8") as f:
            f.write(out.get("next_steps", "") or "")
        print(f"[INFO] Wrote next_steps.txt -> {next_steps_txt_path}")

        logger.info("Run complete. Outputs saved to csv files/ and Text Output files/")

    except Exception as e:
        logger.exception("Pipeline run failed: %s", e)
