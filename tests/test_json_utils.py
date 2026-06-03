"""Unit tests for app.utils.json_utils.extract_json (depth-counting JSON extractor)."""

from app.utils.json_utils import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_surrounded_by_text():
    assert extract_json('here you go: {"a": 1} thanks') == {"a": 1}


def test_nested_objects():
    assert extract_json('{"a": {"b": 2}}') == {"a": {"b": 2}}


def test_brace_inside_string_does_not_close_early():
    # The closing '}' inside the string value must not end the object.
    assert extract_json('{"a": "x}y"}') == {"a": "x}y"}


def test_only_first_object_returned():
    assert extract_json('{"a": 1} {"b": 2}') == {"a": 1}


def test_no_json_returns_none():
    assert extract_json("no json here") is None


def test_malformed_json_returns_none():
    assert extract_json("{not valid}") is None


def test_escaped_quote_in_string():
    assert extract_json(r'{"a": "say \"hi\""}') == {"a": 'say "hi"'}
