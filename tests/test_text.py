import pytest

from openeo_processes_dask.process_implementations.text import *


@pytest.mark.parametrize(
    "string,expected,pattern,case_sensitive",
    [
        ("Lorem ipsum dolor sit amet", False, "amet", True),
        ("Lorem ipsum dolor sit amet", True, "Lorem", True),
        ("Lorem ipsum dolor sit amet", False, "lorem", True),
        ("Lorem ipsum dolor sit amet", True, "lorem", False),
        ("Ä", True, "ä", False),
        (None, "nan", "null", True),
    ],
)
def test_text_begins(string, expected, pattern, case_sensitive):
    result = text_begins(string, pattern, case_sensitive)
    if isinstance(expected, str) and "nan" == expected:
        assert result is None
    else:
        assert result == expected


@pytest.mark.parametrize(
    "string,expected,pattern,case_sensitive",
    [
        ("Lorem ipsum dolor sit amet", True, "amet", True),
        ("Lorem ipsum dolor sit amet", False, "Lorem", True),
        ("Lorem ipsum dolor sit amet", False, "AMET", True),
        ("Lorem ipsum dolor sit amet", True, "AMET", False),
        ("Ä", True, "ä", False),
        (None, "nan", "null", True),
    ],
)
def test_text_ends(string, expected, pattern, case_sensitive):
    result = text_ends(string, pattern, case_sensitive)
    if isinstance(expected, str) and "nan" == expected:
        assert result is None
    else:
        assert result == expected


@pytest.mark.parametrize(
    "string,expected,pattern,case_sensitive",
    [
        ("Lorem ipsum dolor sit amet", False, "openEO", True),
        ("Lorem ipsum dolor sit amet", True, "ipsum dolor", True),
        ("Lorem ipsum dolor sit amet", False, "Ipsum Dolor", True),
        ("Lorem ipsum dolor sit amet", True, "SIT", False),
        ("ÄÖÜ", True, "ö", False),
        (None, "nan", "null", True),
    ],
)
def test_text_contains(string, expected, pattern, case_sensitive):
    result = text_contains(string, pattern, case_sensitive)
    if isinstance(expected, str) and "nan" == expected:
        assert result is None
    else:
        assert result == expected


@pytest.mark.parametrize(
    "data,expected,separator",
    [
        (["Hello", "World"], "Hello World", " "),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 0], "1234567890", ""),
        ([None, True, False, 1, -1.5, "ß"], "none\ntrue\nfalse\n1\n-1.5\nß", "\n"),
        ([2, 0], "210", 1),
        ([], "", ""),
    ],
)
def test_text_contains(data, expected, separator):
    result = text_concat(data, separator)
    assert result == expected
