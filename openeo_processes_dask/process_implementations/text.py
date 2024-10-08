from typing import Any, Optional

__all__ = [
    "text_begins",
    "text_contains",
    "text_concat",
    "text_ends",
]


def text_begins(data: str, pattern: str, case_sensitive: Optional[bool] = True) -> str:
    if data:
        if case_sensitive:
            return data.startswith(pattern)
        else:
            return data.lower().startswith(pattern.lower())
    else:
        return None


def text_contains(
    data: str, pattern: str, case_sensitive: Optional[bool] = True
) -> str:
    if data:
        if case_sensitive:
            return pattern in data
        else:
            return pattern.lower() in data.lower()
    else:
        return None


def text_ends(data: str, pattern: str, case_sensitive: Optional[bool] = True) -> str:
    if data:
        if case_sensitive:
            return data.endswith(pattern)
        else:
            return data.lower().endswith(pattern.lower())
    else:
        return None


def text_concat(data: list[Any], separator: Any) -> str:
    string = ""
    for elem in data:
        if isinstance(elem, bool) or elem is None:
            string += str(elem).lower()
        else:
            string += str(elem)
        if isinstance(separator, bool) or separator is None:
            string += str(separator).lower()
        else:
            string += str(separator)
    if separator == "":
        return string
    else:
        return string[: -len(str(separator))]
