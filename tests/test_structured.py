"""
Tests for Structured Output and JSON Schema enforcement utilities.

python -m pytest tests/test_structured.py -v
"""

from nanochat.structured import (
    repair_json,
    validate_json_schema,
    extract_and_validate_json,
    JSONConstraint,
)


def test_repair_json_clean():
    clean = '{"name": "Alice", "age": 30, "is_active": true}'
    assert repair_json(clean) == {"name": "Alice", "age": 30, "is_active": True}


def test_repair_json_with_markdown_fence():
    fenced = '```json\n{"city": "Tokyo", "temperature": 25.5}\n```'
    assert repair_json(fenced) == {"city": "Tokyo", "temperature": 25.5}


def test_repair_json_trailing_commas():
    trailing = '{"items": [1, 2, 3,], "metadata": {"tag": "v1",},}'
    res = repair_json(trailing)
    assert res == {"items": [1, 2, 3], "metadata": {"tag": "v1"}}


def test_repair_json_unclosed_braces_and_strings():
    # Incomplete generation where generation was truncated mid-way
    incomplete = '{"title": "NanoChat", "tags": ["ai", "transformer"], "config": {"dim": 512'
    res = repair_json(incomplete)
    assert res == {"title": "NanoChat", "tags": ["ai", "transformer"], "config": {"dim": 512}}

    # Truncated inside a string value
    incomplete_str = '{"status": "in_progress", "message": "Downloading shard'
    res_str = repair_json(incomplete_str)
    assert res_str == {"status": "in_progress", "message": "Downloading shard"}


def test_validate_json_schema_types():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "score": {"type": "number"},
            "is_student": {"type": "boolean"},
            "skills": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "age"],
    }

    # Valid data
    valid_data = {
        "name": "Bob",
        "age": 22,
        "score": 95.5,
        "is_student": True,
        "skills": ["python", "pytorch"],
    }
    is_valid, err = validate_json_schema(valid_data, schema)
    assert is_valid
    assert err is None

    # Missing required field
    invalid_missing = {"age": 22}
    is_valid, err = validate_json_schema(invalid_missing, schema)
    assert not is_valid
    assert "Missing required property 'name'" in err

    # Wrong type (boolean where integer expected)
    invalid_type = {"name": "Bob", "age": True}
    is_valid, err = validate_json_schema(invalid_type, schema)
    assert not is_valid
    assert "Expected integer" in err

    # Wrong array item type
    invalid_array = {"name": "Bob", "age": 22, "skills": ["python", 123]}
    is_valid, err = validate_json_schema(invalid_array, schema)
    assert not is_valid
    assert "Expected type 'string'" in err


def test_validate_json_schema_enums():
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["pending", "running", "completed", "failed"]},
        },
        "required": ["status"],
    }

    assert validate_json_schema({"status": "running"}, schema)[0] is True
    assert validate_json_schema({"status": "unknown"}, schema)[0] is False


def test_extract_and_validate_json():
    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
    }

    raw_response = 'Here is your structured result:\n```json\n{"answer": "Paris", "confidence": 0.99}\n```\nHope that helps!'
    data, is_valid, err = extract_and_validate_json(raw_response, schema=schema)
    assert is_valid
    assert data == {"answer": "Paris", "confidence": 0.99}


def test_json_constraint_tracker():
    constraint = JSONConstraint()
    assert not constraint.is_completed

    constraint.update('{"user": "Alice", ')
    assert constraint.depth == 1
    assert not constraint.is_completed

    constraint.update('"friends": ["Bob", "Charlie"]}')
    assert constraint.depth == 0
    assert constraint.is_completed
