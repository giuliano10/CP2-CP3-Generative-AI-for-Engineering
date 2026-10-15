import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import FIELDS, extract_json
from data import generate_examples


def test_generator_is_reproducible_and_valid():
    first = generate_examples(80, seed=42)
    second = generate_examples(80, seed=42)
    assert first == second
    assert len(first) == 80
    for row in first:
        assert set(row["target"]) == set(FIELDS)
        assert isinstance(row["target"]["quantidade"], int)
        assert json.loads(json.dumps(row["target"], ensure_ascii=False)) == row["target"]


def test_parser_accepts_json_and_fenced_json():
    expected = {"produto": "café", "quantidade": 3, "endereco": None, "prazo": "sexta-feira"}
    assert extract_json(json.dumps(expected, ensure_ascii=False)) == expected
    assert extract_json("```json\n" + json.dumps(expected, ensure_ascii=False) + "\n```") == expected
    assert extract_json("sem json") is None
