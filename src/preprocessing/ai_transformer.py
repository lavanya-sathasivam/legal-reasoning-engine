import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.config import (
    COMBINED_CORPUS_PATH,
    CORPUS_MANIFEST_PATH,
    DATA_DIR,
    DEFAULT_CORPUS_MANIFEST,
    EMBEDDINGS_PATH,
    INDEX_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SECTION_INDEX_PATH,
)


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "or", "that", "the", "their", "there",
    "this", "to", "was", "were", "who", "with", "shall", "may", "any", "such",
    "which", "person", "whoever", "said", "under", "than", "not", "into", "where", "when",
}

LAW_ISSUE_HINTS = {
    "BNS": ["criminal"],
    "BNSS": ["procedural"],
    "BSA": ["evidence"],
    "Constitution": ["constitutional"],
}

ISSUE_PATTERNS = {
    "criminal": [
        r"\boffence\b", r"\bpunishment\b", r"\bassault\b", r"\btheft\b", r"\brobbery\b",
        r"\bhurt\b", r"\bmurder\b", r"\bcheating\b", r"\bextortion\b",
    ],
    "procedural": [
        r"\binvestigation\b", r"\barrest\b", r"\bbail\b", r"\btrial\b", r"\bprocedure\b",
        r"\bsearch\b", r"\bseizure\b", r"\bmagistrate\b", r"\bcourt\b",
    ],
    "evidence": [
        r"\bevidence\b", r"\badmissib\w*\b", r"\bproof\b", r"\bwitness\b", r"\bdocument\b",
        r"\btestimony\b", r"\bpresumption\b", r"\bconfession\b",
    ],
    "constitutional": [
        r"\bfundamental right\b", r"\barticle\b", r"\bconstitution\b", r"\bstate\b",
        r"\bequality\b", r"\bliberty\b", r"\bwrit\b", r"\bconstitutional\b",
    ],
}


GENERIC_SECTION_PATTERNS = [
    r"\blimit of punishment\b",
    r"\bdefinitions?\b",
    r"\bgeneral explanations?\b",
    r"\bshort title\b",
    r"\bcommencement\b",
    r"\bclassification of offences\b",
]

CONTEXTUAL_SECTION_PATTERNS = [
    r"\bprivate defence\b",
    r"\bgood faith\b",
    r"\bexceptions?\b",
    r"\babetment\b",
]

TAG_PATTERNS = {
    "assault": [r"\bassault\b", r"\battack\b", r"\bhit\b"],
    "hurt": [r"\bhurt\b", r"\binjur\w*\b", r"\bfracture\b"],
    "theft": [r"\btheft\b", r"\bsteal\w*\b", r"\bstolen\b"],
    "robbery": [r"\brobbery\b", r"\brob\w*\b"],
    "cheating": [r"\bcheat\w*\b", r"\bdecei\w*\b", r"\bfraud\w*\b"],
    "evidence": [r"\bevidence\b", r"\bwitness\b", r"\bdocument\b", r"\bproof\b"],
    "procedure": [r"\barrest\b", r"\bbail\b", r"\binvestigation\b", r"\btrial\b"],
    "rights": [r"\bright\w*\b", r"\bliberty\b", r"\bequality\b"],
}


def ensure_corpus_manifest(manifest_path: Path = CORPUS_MANIFEST_PATH) -> dict[str, Any]:
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CORPUS_MANIFEST, handle, indent=2, ensure_ascii=False)
    return DEFAULT_CORPUS_MANIFEST


def normalize_text(text: str) -> str:
    cleaned = str(text or "").replace("\n", " ").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    cleaned = cleaned.replace("\u2014", " ").replace("\u2013", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z']{2,}", text.lower())
        if token not in STOPWORDS
    ]


def sentence_split(text: str) -> list[str]:
    return [part.strip(" -.;:,") for part in re.split(r"(?<=[.;:])\s+", text) if part.strip()]


def flatten_clause(clause: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    text = normalize_text(clause.get("text", ""))
    if text:
        parts.append(text)
    for child in clause.get("children", []):
        parts.extend(flatten_clause(child))
    return parts


def flatten_section_text(section_record: dict[str, Any]) -> str:
    parts = []
    heading = normalize_text(section_record.get("heading", ""))
    if heading:
        parts.append(heading)
    for clause in section_record.get("clauses", []):
        parts.extend(flatten_clause(clause))
    if not parts:
        parts.extend(
            normalize_text(section_record.get(field, ""))
            for field in ("summary", "original_text", "text")
            if normalize_text(section_record.get(field, ""))
        )
    return normalize_text(" ".join(parts))


def derive_summary(text: str, title: str) -> str:
    sentences = sentence_split(text)
    if sentences:
        first = sentences[0]
        return first if len(first) <= 260 else f"{first[:257].rstrip()}..."
    return title


def derive_key_points(text: str) -> list[str]:
    points = []
    for sentence in sentence_split(text)[:3]:
        if sentence and sentence not in points:
            points.append(sentence)
    return points


def extract_tags(text: str, title: str) -> list[str]:
    combined = f"{title} {text}".lower()
    tags = []
    for tag, patterns in TAG_PATTERNS.items():
        if any(re.search(pattern, combined) for pattern in patterns):
            tags.append(tag)
    counter = Counter(tokenize(combined))
    for token, _ in counter.most_common(8):
        if token not in tags:
            tags.append(token)
    return tags[:12]


def detect_issue_types(text: str, law: str) -> list[str]:
    lowered = text.lower()
    detected = []
    for issue_type, patterns in ISSUE_PATTERNS.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            detected.append(issue_type)
    for fallback_issue in LAW_ISSUE_HINTS.get(law, []):
        if fallback_issue not in detected:
            detected.append(fallback_issue)
    return detected or ["criminal"]


def infer_general_section(title: str, summary: str) -> bool:
    combined = f"{title} {summary}".lower()
    return any(re.search(pattern, combined) for pattern in GENERIC_SECTION_PATTERNS)


def infer_contextual_section(title: str, summary: str) -> bool:
    combined = f"{title} {summary}".lower()
    return any(re.search(pattern, combined) for pattern in CONTEXTUAL_SECTION_PATTERNS)


def build_retrieval_text(record: dict[str, Any]) -> str:
    parts = [
        record["law"],
        record["title"],
        record["summary"],
        " ".join(record["key_points"]),
        " ".join(record["tags"]),
        record["original_text"][:1200],
    ]
    return normalize_text(" ".join(part for part in parts if part))


def _derive_bns_title(section_record: dict[str, Any], section_text: str) -> str:
    heading = normalize_text(section_record.get("heading", ""))
    if heading:
        return heading
    for clause in section_record.get("clauses", []):
        for candidate in flatten_clause(clause):
            if len(candidate) < 12:
                continue
            sentence = sentence_split(candidate)
            if sentence:
                return sentence[0]
    sentences = sentence_split(section_text)
    return sentences[0] if sentences else f"Section {section_record.get('section_number', '')}"


def transform_bns_record(section_record: dict[str, Any], law_config: dict[str, Any]) -> dict[str, Any] | None:
    section_text = flatten_section_text(section_record)
    section = str(section_record.get("section_number", "")).strip()
    if not section or not section_text:
        return None

    law = law_config.get("law_code") or law_config.get("code") or "BNS"
    title = _derive_bns_title(section_record, section_text)
    summary = derive_summary(section_text, title)
    key_points = derive_key_points(section_text)
    tags = extract_tags(section_text, title)
    issue_types = detect_issue_types(f"{title} {section_text}", law)

    record = {
        "law": law,
        "section": section,
        "title": title,
        "summary": summary,
        "key_points": key_points,
        "original_text": section_text,
        "tags": tags,
        "issue_types": issue_types,
        "source_id": f"{law}:{section}",
        "is_general_section": infer_general_section(title, summary),
        "is_contextual_section": infer_contextual_section(title, summary),
    }
    record["retrieval_text"] = build_retrieval_text(record)
    return record


REQUIRED_GENERIC_FIELDS = {
    "law", "section", "title", "summary", "key_points", "original_text", "tags", "issue_types", "source_id",
}


def normalize_generic_record(record: dict[str, Any], law_config: dict[str, Any]) -> dict[str, Any] | None:
    if not REQUIRED_GENERIC_FIELDS.issubset(record):
        return None

    normalized = {
        "law": normalize_text(record["law"]),
        "section": normalize_text(record["section"]),
        "title": normalize_text(record["title"]),
        "summary": normalize_text(record["summary"]),
        "key_points": [normalize_text(item) for item in record.get("key_points", []) if normalize_text(item)],
        "original_text": normalize_text(record.get("original_text", "")),
        "tags": [normalize_text(item).lower() for item in record.get("tags", []) if normalize_text(item)],
        "issue_types": [normalize_text(item).lower() for item in record.get("issue_types", []) if normalize_text(item)],
        "source_id": normalize_text(record["source_id"]),
        "is_general_section": bool(record.get("is_general_section", False)),
        "is_contextual_section": bool(record.get("is_contextual_section", False)),
    }
    if not normalized["law"]:
        normalized["law"] = law_config.get("law_code", "")
    if not normalized["summary"]:
        normalized["summary"] = derive_summary(normalized["original_text"], normalized["title"])
    normalized["retrieval_text"] = build_retrieval_text(normalized)
    return normalized


def _resolve_data_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else DATA_DIR / path


def _load_bns_source(raw_path: Path) -> dict[str, Any]:
    candidates = [raw_path, RAW_DIR / "corrected_bns.json", Path("corrected_bns.json")]
    for candidate in candidates:
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError("BNS raw JSON not found in expected locations.")


def process_law_dataset(law_code: str, law_config: dict[str, Any]) -> list[dict[str, Any]]:
    loader = law_config.get("loader", "generic_records")
    processed_path = _resolve_data_path(law_config.get("processed_path"))
    raw_path = _resolve_data_path(law_config.get("raw_path"))
    law_config = {**law_config, "law_code": law_code}

    if loader == "bns_json":
        source = _load_bns_source(raw_path or RAW_DIR / "corrected_bns.json")
        records = [
            transformed
            for item in source.get("sections", [])
            if (transformed := transform_bns_record(item, law_config)) is not None
        ]
    elif loader == "generic_records":
        if not raw_path or not raw_path.exists():
            return []
        with raw_path.open("r", encoding="utf-8") as handle:
            source = json.load(handle)
        entries = source if isinstance(source, list) else source.get("records", [])
        records = [
            normalized
            for item in entries
            if (normalized := normalize_generic_record(item, law_config)) is not None
        ]
    else:
        raise ValueError(f"Unsupported loader '{loader}' for law {law_code}.")

    if processed_path:
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        with processed_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)
    return records


def build_corpus(
    manifest_path: Path = CORPUS_MANIFEST_PATH,
    output_path: Path = COMBINED_CORPUS_PATH,
    index_path: Path = SECTION_INDEX_PATH,
) -> list[dict[str, Any]]:
    manifest = ensure_corpus_manifest(manifest_path)
    enabled_laws = manifest.get("enabled_laws", [])
    law_configs = manifest.get("laws", {})

    corpus: list[dict[str, Any]] = []
    for law_code in enabled_laws:
        law_config = law_configs.get(law_code, {})
        corpus.extend(process_law_dataset(law_code, law_config))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(corpus, handle, indent=2, ensure_ascii=False)

    section_id_map = {
        str(index): {
            "source_id": item["source_id"],
            "law": item["law"],
            "section": item["section"],
            "title": item["title"],
        }
        for index, item in enumerate(corpus)
    }
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(section_id_map, handle, indent=2, ensure_ascii=False)

    fallback_embeddings_path = EMBEDDINGS_PATH.with_suffix(".json")
    if EMBEDDINGS_PATH.exists():
        EMBEDDINGS_PATH.unlink()
    if fallback_embeddings_path.exists():
        fallback_embeddings_path.unlink()

    return corpus


def transform_bns_file(raw_path: Path | None = None, output_path: Path | None = None) -> list[dict[str, Any]]:
    manifest = ensure_corpus_manifest()
    manifest = dict(manifest)
    manifest["enabled_laws"] = ["BNS"]
    manifest["laws"] = dict(manifest.get("laws", {}))
    manifest["laws"]["BNS"] = dict(manifest["laws"].get("BNS", {}))
    if raw_path is not None:
        manifest["laws"]["BNS"]["raw_path"] = str(raw_path)
    if output_path is not None:
        manifest["laws"]["BNS"]["processed_path"] = str(output_path)

    temp_manifest = PROCESSED_DIR / "_bns_only_manifest.json"
    with temp_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    records = build_corpus(manifest_path=temp_manifest)
    temp_manifest.unlink(missing_ok=True)
    return records


if __name__ == "__main__":
    transformed = build_corpus()
    print(f"Generated {len(transformed)} legal corpus records at {COMBINED_CORPUS_PATH}.")
