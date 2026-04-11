from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"

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

DEFAULT_CORPUS_MANIFEST = {
    "enabled_laws": ["BNS"],
    "laws": {
        "BNS": {
            "label": "Bharatiya Nyaya Sanhita",
            "loader": "bns_json",
            "raw_path": "raw/corrected_bns.json",
            "processed_path": "processed/bns_sections_ai_ready.json",
        },
        "BNSS": {
            "label": "Bharatiya Nagarik Suraksha Sanhita",
            "loader": "generic_records",
            "raw_path": "raw/bnss/sections.json",
            "processed_path": "processed/bnss_sections_ai_ready.json",
        },
        "BSA": {
            "label": "Bharatiya Sakshya Adhiniyam",
            "loader": "generic_records",
            "raw_path": "raw/bsa/sections.json",
            "processed_path": "processed/bsa_sections_ai_ready.json",
        },
        "Constitution": {
            "label": "Constitution of India",
            "loader": "generic_records",
            "raw_path": "raw/constitution/sections.json",
            "processed_path": "processed/constitution_sections_ai_ready.json",
        },
    },
}
