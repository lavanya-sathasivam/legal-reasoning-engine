from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
SCHEMA_DIR = DATA_DIR / "schemas"
APP_DB_PATH = DATA_DIR / "platform.db"

CORPUS_MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
COMBINED_CORPUS_PATH = PROCESSED_DIR / "legal_corpus.json"
EMBEDDINGS_PATH = PROCESSED_DIR / "legal_corpus_embeddings.npy"
SECTION_INDEX_PATH = INDEX_DIR / "section_id_map.json"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K_RESULTS = 5
DEFAULT_SEARCH_CANDIDATES = 12
MIN_RETRIEVAL_SCORE = 0.36
MIN_LEGAL_CONFIDENCE = 0.43
DEFAULT_LAWS = ("BNS", "BNSS", "BSA", "Constitution")
LEGAL_SECTION_SCHEMA_VERSION = "2.0"

DEFAULT_CORPUS_MANIFEST = {
    "schema_version": LEGAL_SECTION_SCHEMA_VERSION,
    "enabled_laws": ["BNS", "BNSS", "BSA", "Constitution"],
    "laws": {
        "BNS": {
            "label": "Bharatiya Nyaya Sanhita",
            "loader": "bns_json",
            "raw_path": "raw/corrected_bns.json",
            "processed_path": "processed/bns_sections_ai_ready.json",
            "status": "raw_imported",
            "source_type": "json",
        },
        "BNSS": {
            "label": "Bharatiya Nagarik Suraksha Sanhita",
            "loader": "structured_act_json",
            "raw_path": "../corrected_bnss.json",
            "processed_path": "processed/bnss_sections_ai_ready.json",
            "status": "raw_imported",
            "source_type": "json",
        },
        "BSA": {
            "label": "Bharatiya Sakshya Adhiniyam",
            "loader": "structured_act_json",
            "raw_path": "../corrected_bsa.json",
            "processed_path": "processed/bsa_sections_ai_ready.json",
            "status": "raw_imported",
            "source_type": "json",
        },
        "Constitution": {
            "label": "Constitution of India",
            "loader": "structured_act_json",
            "raw_path": "../constitution_en.json",
            "processed_path": "processed/constitution_sections_ai_ready.json",
            "status": "raw_imported",
            "source_type": "json",
        },
    },
}
