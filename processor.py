import os
import time
import fitz  # PyMuPDF
import docx
import requests
from typing import List, Dict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import logging

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ---------------------------
# HTTP Session with Rate Limiting
# ---------------------------
def create_session(max_retries=3, backoff_factor=0.5) -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = create_session()


# ---------------------------
# File Parsing & Chunking
# ---------------------------
def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = "\n".join([page.get_text() for page in doc])
        logger.info(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise RuntimeError(f"Failed to extract text from PDF: {e}")


def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        logger.info(f"Extracted {len(text)} characters from DOCX.")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from DOCX: {e}")
        raise RuntimeError(f"Failed to extract text from DOCX: {e}")


def chunk_text(text: str, max_words: int = 1500) -> List[str]:
    try:
        words = text.split()
        chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
        logger.info(f"Text chunked into {len(chunks)} parts.")
        return chunks
    except Exception as e:
        logger.error(f"Failed to chunk text: {e}")
        raise RuntimeError(f"Failed to chunk text: {e}")


# ---------------------------
# Gemini API Integration
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class GeminiChatSession:
    def __init__(self, model="gemini-1.5-flash"):
        self.api_key = GEMINI_API_KEY
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        self.headers = {"Content-Type": "application/json"}
        self.history = []

        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in environment variables.")

    def send_message(self, prompt: str, retry: int = 3) -> str:
        self.history.append({"role": "user", "parts": [{"text": prompt}]})
        payload = {"contents": self.history}

        for attempt in range(retry):
            try:
                response = requests.post(
                    f"{self.api_url}?key={self.api_key}",
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                response_text = data["candidates"][0]["content"]["parts"][0]["text"]
                self.history.append({"role": "model", "parts": [{"text": response_text}]})
                return response_text
            except Exception as e:
                if attempt < retry - 1:
                    logger.warning(f"Retrying due to error: {e} (Attempt {attempt + 1})")
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Gemini chat failed after {retry} attempts: {e}")
                    raise RuntimeError(f"Gemini chat failed after {retry} attempts: {e}")


def extract_features_from_chunk(chat_session: GeminiChatSession, chunk: str) -> Dict:
    try:
        prompt = f"""
        Extract the following information:
        - Project Area GPS coordinates (polygon preferred)
        - Project Start Date
        - Project Completion Date
        - Vegetation info
        - Forestation info
        - AGB (Aboveground Biomass)
        - Reported Carbon Offset

        Text:
        {chunk}
        """
        response_text = chat_session.send_message(prompt)
        return {"raw_response": response_text}
    except Exception as e:
        logger.error(f"Failed to extract features from Gemini: {e}")
        raise RuntimeError(f"Failed to extract features from Gemini: {e}")


def process_chunks(file_text: str, chunk_size: int = 1500) -> List[Dict]:
    chunks = [file_text[i:i + chunk_size] for i in range(0, len(file_text), chunk_size)]
    chat_session = GeminiChatSession()
    extracted_features = []
    for chunk in chunks:
        features = extract_features_from_chunk(chat_session, chunk)
        extracted_features.append(features)
    logger.info(f"Processed {len(extracted_features)} chunks.")
    return extracted_features


# ---------------------------
# Data Aggregation Logic
# ---------------------------
def merge_features(results: List[Dict]) -> Dict:
    try:
        merged = {"gps": None, "start_date": None, "end_date": None, "reported_offset": None, "area": None}
        for result in results:
            text = result.get("raw_response", "")
            if "gps coordinates" in text.lower():
                merged["gps"] = extract_gps_from_text(text)
            if "start date" in text.lower():
                merged["start_date"] = extract_date_from_text(text, "start")
            if "end date" in text.lower():
                merged["end_date"] = extract_date_from_text(text, "end")
            if "carbon offset" in text.lower():
                merged["reported_offset"] = extract_reported_offset(text)
            if "area" in text.lower():
                merged["area"] = extract_area_from_text(text)
        logger.info(f"Merged extracted features: {merged}")
        return merged
    except Exception as e:
        logger.error(f"Failed to merge extracted features: {e}")
        raise RuntimeError(f"Failed to merge extracted features: {e}")


# ---------------------------
# Helper Functions
# ---------------------------
def extract_gps_from_text(text: str) -> List[List[float]]:
    return [[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]


def extract_date_from_text(text: str, date_type: str) -> str:
    date_pattern = r"\b\d{4}-\d{2}-\d{2}\b"
    match = re.search(date_pattern, text)
    return match.group(0) if match else "Unknown"


def extract_reported_offset(text: str) -> float:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*tons", text)
    return float(match.group(1)) if match else 0.0


def extract_area_from_text(text: str) -> float:
    match = re.search(r"(\d+(\.\d+)?)\s*hectares", text)
    return float(match.group(1)) if match else 0.0


# ---------------------------
# Carbon Offset Calculation
# ---------------------------
def calculate_carbon_offset(agb_before: float, agb_after: float, area_hectares: float) -> float:
    try:
        delta_agb = max(0, agb_after - agb_before)
        return round(0.47 * delta_agb * area_hectares, 2)
    except Exception as e:
        logger.error(f"Failed to calculate carbon offset: {e}")
        return 0.0


# ---------------------------
# Final Orchestrator Function
# ---------------------------
def process_project_report(file_path: str) -> Dict:
    try:
        if file_path.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_path.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        else:
            raise ValueError("Unsupported file type")

        features = process_chunks(text)
        merged = merge_features(features)

        area = merged["area"] or 10.0
        agb_before = 100
        agb_after = 125

        offset = calculate_carbon_offset(agb_before, agb_after, area)
        merged["calculated_offset"] = offset

        logger.info(f"Final processed report data: {merged}")
        return merged

    except Exception as e:
        logger.error(f"Error in processing report: {e}")
        return {"error": str(e)}