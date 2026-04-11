import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from google import genai
from tqdm import tqdm


DEFAULT_INPUT_PATH = Path("bns_en.json")
DEFAULT_OUTPUT_PATH = Path("corrected_bns.json")
DEFAULT_CHUNK_SIZE = 25
DEFAULT_RETRIES = 5
DEFAULT_SLEEP_SECONDS = 3
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 60
MODEL_NAME = "gemini-2.5-flash"


def chunk_data(data: dict[str, Any], size: int = DEFAULT_CHUNK_SIZE):
    sections = data["sections"]
    for index in range(0, len(sections), size):
        yield sections[index:index + size]


def clean_response(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def validate_json(text: str) -> list[dict[str, Any]] | None:
    try:
        parsed = json.loads(clean_response(text))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, list) else None


def build_client(api_key: str | None = None) -> genai.Client:
    resolved_key = api_key or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise RuntimeError("GOOGLE_API_KEY is required to run the OCR correction parser.")
    return genai.Client(api_key=resolved_key)


def correct_chunk(client: genai.Client, chunk: list[dict[str, Any]]) -> str:
    prompt = f"""
You are correcting OCR errors in Bharatiya Nyaya Sanhita JSON.

Rules:
- Fix merged words (e.g., "Inthis" -> "In this")
- Preserve EXACT legal meaning
- DO NOT change structure
- Keep all nested fields like children
- Output ONLY JSON array

INPUT:
{json.dumps(chunk, ensure_ascii=False)}
"""

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    return response.text


def process_file(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    retries: int = DEFAULT_RETRIES,
    sleep_seconds: int = DEFAULT_SLEEP_SECONDS,
    rate_limit_wait_seconds: int = DEFAULT_RATE_LIMIT_WAIT_SECONDS,
    api_key: str | None = None,
) -> dict[str, Any]:
    client = build_client(api_key=api_key)

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    corrected_sections: list[dict[str, Any]] = []
    chunks = list(chunk_data(data, chunk_size))

    for chunk in tqdm(chunks):
        success = False

        for _ in range(retries):
            try:
                result = correct_chunk(client, chunk)
                valid = validate_json(result)
                if valid is not None:
                    corrected_sections.extend(valid)
                    success = True
                    break
                print("Invalid JSON response, retrying...")
            except Exception as exc:
                message = str(exc)
                if "429" in message:
                    print(f"Rate limit hit. Waiting {rate_limit_wait_seconds} seconds...")
                    time.sleep(rate_limit_wait_seconds)
                else:
                    print(f"Parser request failed: {message}")
                    time.sleep(5)

        if not success:
            print("Failed chunk, keeping original input for that chunk.")
            corrected_sections.extend(chunk)

        time.sleep(sleep_seconds)

    final_output = {
        "act_id": data["act_id"],
        "language": data["language"],
        "sections": corrected_sections,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(final_output, handle, indent=2, ensure_ascii=False)

    return final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correct OCR issues in BNS JSON using Gemini.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Source JSON path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Corrected JSON output path")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Sections per request")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Retries per chunk")
    parser.add_argument("--sleep-seconds", type=int, default=DEFAULT_SLEEP_SECONDS, help="Delay between requests")
    parser.add_argument(
        "--rate-limit-wait-seconds",
        type=int,
        default=DEFAULT_RATE_LIMIT_WAIT_SECONDS,
        help="Wait time after a 429 response",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = process_file(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        rate_limit_wait_seconds=args.rate_limit_wait_seconds,
    )
    print(f"Done. Wrote {len(output['sections'])} corrected sections to {args.output}.")


if __name__ == "__main__":
    main()
