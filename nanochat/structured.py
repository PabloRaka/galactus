"""
Structured Output and Schema Enforcement (JSON Mode) for Galactus/Nanochat.

Provides:
- JSON extraction and robust parsing with auto-repair
- Standard JSON Schema validator (properties, types, required, enums, arrays)
- Token-level JSON constraint state tracker for autoregressive decoding
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Union


def repair_json(text: str) -> Optional[Any]:
    """
    Attempt to repair incomplete, truncated, or slightly malformed JSON strings.
    Handles:
    - Markdown code fences (```json ... ``` embedded anywhere)
    - Trailing commentary outside the JSON object
    - Trailing commas before closing braces/brackets
    - Unclosed quotes
    - Unclosed braces '{' and brackets '['
    """
    if not text or not isinstance(text, str):
        return None

    s = text.strip()

    # Fast path 1: try standard parsing directly
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fast path 2: extract markdown code fence if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if fence_match:
        fence_content = fence_match.group(1).strip()
        try:
            return json.loads(fence_content)
        except json.JSONDecodeError:
            s = fence_content

    # Extract JSON candidate bounded by first { or [
    first_brace = s.find('{')
    first_bracket = s.find('[')

    if first_brace == -1 and first_bracket == -1:
        return None

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx = first_brace
    else:
        start_idx = first_bracket

    candidate = s[start_idx:]

    # Remove trailing commas
    candidate = re.sub(r',\s*([\]\}])', r'\1', candidate)

    # Check for unclosed string literal and track matching braces/brackets
    in_string = False
    escape = False
    stack = []
    found_end_idx = -1

    for i, char in enumerate(candidate):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char in '{[':
                stack.append('}' if char == '{' else ']')
            elif char in '}]':
                if stack and stack[-1] == char:
                    stack.pop()
                    if not stack:
                        found_end_idx = i + 1
                        break

    # If a complete root object/array was cleanly closed, use that slice
    if found_end_idx != -1:
        clean_candidate = candidate[:found_end_idx]
        try:
            return json.loads(clean_candidate)
        except json.JSONDecodeError:
            candidate = clean_candidate

    # Otherwise, it was truncated. Repair unclosed string and unclosed stack.
    if in_string:
        candidate += '"'

    candidate = re.sub(r',\s*$', '', candidate)

    while stack:
        candidate += stack.pop()

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def validate_json_schema(data: Any, schema: Dict[str, Any], path: str = "$") -> Tuple[bool, Optional[str]]:
    """
    Validate a parsed Python object against a JSON schema specification.
    Supports:
    - type: "object", "array", "string", "number", "integer", "boolean", "null"
    - properties & required (for objects)
    - items (for arrays)
    - enum (for allowed values)
    """
    if not schema or not isinstance(schema, dict):
        return True, None

    expected_type = schema.get("type")
    if expected_type:
        type_mapping = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }
        # Note: In Python, bool is a subclass of int, so handle carefully
        if expected_type == "integer" and isinstance(data, bool):
            return False, f"Expected integer at {path}, got boolean {data}"
        if expected_type == "number" and isinstance(data, bool):
            return False, f"Expected number at {path}, got boolean {data}"

        py_type = type_mapping.get(expected_type)
        if py_type and not isinstance(data, py_type):
            return False, f"Expected type '{expected_type}' at {path}, got '{type(data).__name__}'"

    # Validate enum
    if "enum" in schema and schema["enum"]:
        if data not in schema["enum"]:
            return False, f"Value at {path} ({data!r}) is not in allowed enum: {schema['enum']}"

    # Validate Object Properties & Required Fields
    if isinstance(data, dict):
        required_keys = schema.get("required", [])
        for req_key in required_keys:
            if req_key not in data:
                return False, f"Missing required property '{req_key}' at {path}"

        properties = schema.get("properties", {})
        for prop_name, prop_val in data.items():
            if prop_name in properties:
                prop_schema = properties[prop_name]
                valid, err = validate_json_schema(prop_val, prop_schema, path=f"{path}.{prop_name}")
                if not valid:
                    return False, err

    # Validate Array Items
    if isinstance(data, list) and "items" in schema:
        item_schema = schema["items"]
        for idx, item in enumerate(data):
            valid, err = validate_json_schema(item, item_schema, path=f"{path}[{idx}]")
            if not valid:
                return False, err

    return True, None


def extract_and_validate_json(text: str, schema: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], bool, Optional[str]]:
    """
    Extracts JSON from text, repairs if needed, and validates against an optional schema.
    Returns: (parsed_data, is_valid, error_message)
    """
    parsed = repair_json(text)
    if parsed is None:
        return None, False, "Failed to parse valid JSON from text"

    if schema is not None:
        is_valid, err = validate_json_schema(parsed, schema)
        if not is_valid:
            return parsed, False, err

    return parsed, True, None


class JSONConstraint:
    """
    State tracker for autoregressive generation in JSON mode.
    Monitors depth of objects/arrays and can signal early completion
    once a root JSON structure has been closed.
    """

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema
        self.depth = 0
        self.has_started = False
        self.in_string = False
        self.escape = False
        self.is_completed = False

    def update(self, char_text: str):
        """Update tracker state with newly generated character/text sequence."""
        for char in char_text:
            if self.escape:
                self.escape = False
                continue
            if char == '\\':
                self.escape = True
                continue
            if char == '"':
                self.in_string = not self.in_string
                continue
            if not self.in_string:
                if char in '{[':
                    self.depth += 1
                    self.has_started = True
                elif char in '}]':
                    if self.depth > 0:
                        self.depth -= 1
                        if self.depth == 0 and self.has_started:
                            self.is_completed = True
