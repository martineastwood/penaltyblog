import re
from datetime import date, datetime
from typing import Any, Union

from .predicates import AndPredicate, FieldPredicate, NotPredicate, OrPredicate


def _ensure_scalar_or_raise(v, field: str):
    if isinstance(v, dict):
        raise TypeError(f"Cannot apply comparison on dict field '{field}'")
    if isinstance(v, list) and any(isinstance(i, dict) for i in v):
        raise TypeError(f"Cannot apply comparison on list of dicts: '{field}'")
    return v


def _ensure_comparable_or_raise(v, field: str, threshold):
    """Ensure that v and threshold are comparable types."""
    if isinstance(v, dict):
        raise TypeError(f"Cannot apply comparison on dict field '{field}'")
    if isinstance(v, list) and any(isinstance(i, dict) for i in v):
        raise TypeError(f"Cannot apply comparison on list of dicts: '{field}'")

    # Check if types are comparable
    if v is None:
        return False  # None values shouldn't match comparisons

    # Handle date/datetime compatibility specially
    if isinstance(v, (date, datetime)) and isinstance(threshold, (date, datetime)):
        return True

    # Allow comparisons between compatible types
    compatible_pairs = [
        ((int, float), (int, float)),  # numeric types
        ((str,), (str,)),  # strings
        ((bool,), (bool,)),  # booleans
    ]

    v_type = type(v)
    threshold_type = type(threshold)

    for type_group1, type_group2 in compatible_pairs:
        if isinstance(v, type_group1) and isinstance(threshold, type_group2):
            return True

    raise TypeError(
        f"Cannot compare field '{field}' of type {v_type.__name__} "
        f"with value of type {threshold_type.__name__}"
    )


def _normalize_for_comparison(v, threshold):
    """Normalize values for comparison, handling date/datetime conversion."""
    # Handle date/datetime compatibility
    if isinstance(v, (date, datetime)) and isinstance(threshold, (date, datetime)):
        # Convert date to datetime for comparison if needed
        v_norm = (
            v if isinstance(v, datetime) else datetime.combine(v, datetime.min.time())
        )
        threshold_norm = (
            threshold
            if isinstance(threshold, datetime)
            else datetime.combine(threshold, datetime.min.time())
        )
        return v_norm, threshold_norm

    # For other types, return as-is
    return v, threshold


# === Field comparison helpers ===
def where_equals(field: str, value: Any):
    return FieldPredicate(field, lambda v: _ensure_scalar_or_raise(v, field) == value)


def where_not_equals(field: str, value: Any):
    return FieldPredicate(field, lambda v: _ensure_scalar_or_raise(v, field) != value)


def where_in(field: str, values: list):
    def test(v):
        if isinstance(v, dict):
            raise TypeError(f"Cannot use `where_in` on dict field '{field}'")
        if isinstance(v, list):
            if any(isinstance(i, dict) for i in v):
                raise TypeError(f"Cannot use `where_in` on list of dicts: '{field}'")
            return any(item in values for item in v)
        return v in values

    return FieldPredicate(field, test)


def where_not_in(field: str, values: list):
    def test(v):
        if isinstance(v, dict):
            raise TypeError(f"Cannot use `where_not_in` on dict field '{field}'")
        if isinstance(v, list):
            if any(isinstance(i, dict) for i in v):
                raise TypeError(
                    f"Cannot use `where_not_in` on list of dicts: '{field}'"
                )
            return all(item not in values for item in v)
        return v not in values

    return FieldPredicate(field, test)


def where_contains(field: str, substring: str):
    return FieldPredicate(
        field, lambda v: substring in (str(v) if v is not None else "")
    )


def where_startswith(field: str, prefix: str):
    return FieldPredicate(
        field, lambda v: (str(v) if v is not None else "").startswith(prefix)
    )


def where_endswith(field: str, suffix: str):
    return FieldPredicate(
        field, lambda v: (str(v) if v is not None else "").endswith(suffix)
    )


def where_exists(field: str):
    return FieldPredicate(field, lambda v: v is not None)


def where_is_null(field: str):
    return FieldPredicate(field, lambda v: v is None)


def where_gt(field: str, threshold):
    def compare(v):
        if not _ensure_comparable_or_raise(v, field, threshold):
            return False
        v_normalized, threshold_normalized = _normalize_for_comparison(v, threshold)
        return v_normalized > threshold_normalized

    return FieldPredicate(field, compare)


def where_gte(field: str, threshold):
    def compare(v):
        if not _ensure_comparable_or_raise(v, field, threshold):
            return False
        v_normalized, threshold_normalized = _normalize_for_comparison(v, threshold)
        return v_normalized >= threshold_normalized

    return FieldPredicate(field, compare)


def where_lt(field: str, threshold):
    def compare(v):
        if not _ensure_comparable_or_raise(v, field, threshold):
            return False
        v_normalized, threshold_normalized = _normalize_for_comparison(v, threshold)
        return v_normalized < threshold_normalized

    return FieldPredicate(field, compare)


def where_lte(field: str, threshold):
    def compare(v):
        if not _ensure_comparable_or_raise(v, field, threshold):
            return False
        v_normalized, threshold_normalized = _normalize_for_comparison(v, threshold)
        return v_normalized <= threshold_normalized

    return FieldPredicate(field, compare)


# === Regex matching ===
def where_regex_match(field: str, pattern: str, flags: Union[int, re.RegexFlag] = 0):
    """
    Create a predicate that tests if a field matches a regex pattern.

    Args:
        field (str): The field to check.
        pattern (str): The regex pattern to match against.
        flags (int or re.RegexFlag, optional): Regex flags (e.g., re.IGNORECASE).

    Returns:
        FieldPredicate: A predicate that tests if the field matches the pattern.
    """
    try:
        # Compile the pattern once for efficiency
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern '{pattern}': {e}")

    def test(v):
        if v is None:
            return False
        try:
            # Convert to string if not already
            v_str = str(v) if not isinstance(v, str) else v
            return bool(compiled_pattern.search(v_str))
        except (TypeError, ValueError):
            return False

    return FieldPredicate(field, test)


# === Combinators ===
def and_(*preds):
    return AndPredicate(*preds)


def or_(*preds):
    return OrPredicate(*preds)


def not_(pred):
    return NotPredicate(pred)
