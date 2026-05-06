import json
import tempfile
import unittest
from pathlib import Path

from src.preprocessing.ai_transformer import build_corpus, ensure_corpus_manifest, transform_bns_file


class PreprocessingTests(unittest.TestCase):
    def test_default_manifest_is_created_with_enabled_bns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "corpus_manifest.json"
            manifest = ensure_corpus_manifest(manifest_path)
            self.assertIn("BNS", manifest["enabled_laws"])
            self.assertIn("BNSS", manifest["enabled_laws"])
            self.assertIn("BNSS", manifest["laws"])
            self.assertTrue(manifest_path.exists())

    def test_bns_transform_produces_unified_record_fields(self) -> None:
        records = transform_bns_file()
        self.assertTrue(records)
        sample = records[0]
        expected_fields = {
            "law",
            "section",
            "title",
            "summary",
            "key_points",
            "original_text",
            "tags",
            "issue_types",
            "source_id",
            "retrieval_text",
        }
        self.assertTrue(expected_fields.issubset(sample.keys()))
        self.assertEqual(sample["source_id"], f"{sample['law']}:{sample['section']}")

    def test_build_corpus_supports_manifest_driven_multilaw_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_dir = root / "data"
            raw_dir = data_dir / "raw"
            processed_dir = data_dir / "processed"
            index_dir = data_dir / "indexes"
            raw_dir.mkdir(parents=True)
            processed_dir.mkdir(parents=True)
            index_dir.mkdir(parents=True)

            bnss_records = [{
                "law": "BNSS",
                "section": "41",
                "title": "Arrest procedure",
                "summary": "Procedure for arrest by police.",
                "key_points": ["Covers arrest steps", "Mentions police procedure"],
                "original_text": "Procedure for arrest by police officers.",
                "tags": ["arrest", "procedure"],
                "issue_types": ["procedural"],
                "source_id": "BNSS:41",
            }]
            bsa_records = [{
                "law": "BSA",
                "section": "12",
                "title": "Electronic evidence",
                "summary": "Admissibility of electronic records.",
                "key_points": ["Electronic records may be evidence"],
                "original_text": "Electronic records can be admitted as evidence.",
                "tags": ["evidence", "electronic"],
                "issue_types": ["evidence"],
                "source_id": "BSA:12",
            }]
            constitution_records = [{
                "law": "Constitution",
                "section": "21",
                "title": "Protection of life and personal liberty",
                "summary": "No person shall be deprived of life or personal liberty except according to procedure established by law.",
                "key_points": ["Protects life and liberty"],
                "original_text": "Article 21 protects life and personal liberty.",
                "tags": ["liberty", "rights"],
                "issue_types": ["constitutional"],
                "source_id": "Constitution:21",
            }]

            (raw_dir / "bnss.json").write_text(json.dumps(bnss_records), encoding="utf-8")
            (raw_dir / "bsa.json").write_text(json.dumps(bsa_records), encoding="utf-8")
            (raw_dir / "constitution.json").write_text(json.dumps(constitution_records), encoding="utf-8")

            manifest = {
                "enabled_laws": ["BNSS", "BSA", "Constitution"],
                "laws": {
                    "BNSS": {
                        "loader": "generic_records",
                        "raw_path": str(raw_dir / "bnss.json"),
                        "processed_path": str(processed_dir / "bnss.json"),
                    },
                    "BSA": {
                        "loader": "generic_records",
                        "raw_path": str(raw_dir / "bsa.json"),
                        "processed_path": str(processed_dir / "bsa.json"),
                    },
                    "Constitution": {
                        "loader": "generic_records",
                        "raw_path": str(raw_dir / "constitution.json"),
                        "processed_path": str(processed_dir / "constitution.json"),
                    },
                },
            }
            manifest_path = data_dir / "corpus_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            combined_output = processed_dir / "combined.json"
            index_output = index_dir / "section_map.json"
            corpus = build_corpus(manifest_path=manifest_path, output_path=combined_output, index_path=index_output)

            self.assertEqual({item["law"] for item in corpus}, {"BNSS", "BSA", "Constitution"})
            self.assertTrue(all(item["source_id"] for item in corpus))
            self.assertTrue(combined_output.exists())
            self.assertTrue(index_output.exists())


if __name__ == "__main__":
    unittest.main()
