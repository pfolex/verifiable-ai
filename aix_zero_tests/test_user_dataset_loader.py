"""Unit tests for user JSONL dataset provenance, context-fit, and split isolation (`#440`)."""
import json
from pathlib import Path
import tempfile
import unittest

from aix_zero.user_dataset_loader import load_and_validate_user_jsonl_dataset


class UserDatasetLoaderTests(unittest.TestCase):
    def test_validates_provenance_roles_overlength_and_duplicates(self):
        rows = [
            # Valid train row
            {"task_id": "t1", "messages": [{"role": "user", "content": "2+2?"}], "target": "4", "source": "local", "license": "MIT", "task_family": "math", "split": "train"},
            # Valid dev row
            {"task_id": "d1", "messages": [{"role": "user", "content": "3+3?"}], "target": "6", "source": "local", "license": "MIT", "task_family": "math", "split": "development"},
            # Missing provenance
            {"task_id": "bad_prov", "messages": [{"role": "user", "content": "1+1?"}], "target": "2", "source": "", "license": "MIT", "task_family": "math", "split": "train"},
            # Unsupported role
            {"task_id": "bad_role", "messages": [{"role": "hacker", "content": "hi"}], "target": "ok", "source": "local", "license": "MIT", "task_family": "chat", "split": "train"},
            # Overlength target
            {"task_id": "too_long", "messages": [{"role": "user", "content": "x" * 100}], "target": "y" * 100, "source": "local", "license": "MIT", "task_family": "chat", "split": "train"},
            # Intra-split duplicate
            {"task_id": "dup_t1", "messages": [{"role": "user", "content": "2+2?"}], "target": "4", "source": "local", "license": "MIT", "task_family": "math", "split": "train"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            mpath = Path(tmp) / "manifest.json"
            res = load_and_validate_user_jsonl_dataset(p, max_seq_len=64, manifest_output_path=mpath)
            m = res["manifest"]
            self.assertTrue(mpath.exists())
            self.assertEqual(m["split_counts"]["train"], 1)
            self.assertEqual(m["split_counts"]["development"], 1)
            self.assertEqual(m["exclusion_counts"]["missing_provenance"], 1)
            self.assertEqual(m["exclusion_counts"]["unsupported_role"], 1)
            self.assertEqual(m["exclusion_counts"]["overlength_target"], 1)
            self.assertEqual(m["exclusion_counts"]["intra_split_duplicate"], 1)

    def test_rejects_cross_split_prompt_leakage(self):
        rows = [
            {"task_id": "t1", "prompt": "def f(): return 1", "target": "1", "test_code": "assert f() == 1", "source": "repo", "license": "Apache-2.0", "task_family": "python", "split": "train"},
            {"task_id": "e1", "prompt": "def f(): return 1", "target": "1", "test_code": "assert f() == 1", "source": "repo", "license": "Apache-2.0", "task_family": "python", "split": "reserved_final"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "code.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Cross-split prompt leakage"):
                load_and_validate_user_jsonl_dataset(p, task_format="code_task", max_seq_len=128)


if __name__ == "__main__":
    unittest.main()
