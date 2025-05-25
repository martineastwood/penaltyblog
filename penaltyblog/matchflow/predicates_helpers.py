from typing import Any, List

from .predicates import AndPredicate, FieldPredicate, NotPredicate, OrPredicate


def _ensure_scalar_or_raise(v, field: str):
    if isinstance(v, dict):
        raise TypeError(f"Cannot apply comparison on dict field '{field}'")
    if isinstance(v, list) and any(isinstance(i, dict) for i in v):
        raise TypeError(f"Cannot apply comparison on list of dicts: '{field}'")
    return v


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


def where_exists(field: str):
    return FieldPredicate(field, lambda v: v is not None)


def where_is_null(field: str):
    return FieldPredicate(field, lambda v: v is None)


def where_gt(field: str, threshold: float):
    return FieldPredicate(
        field,
        lambda v: isinstance(_ensure_scalar_or_raise(v, field), (int, float))
        and v > threshold,
    )


def where_gte(field: str, threshold: float):
    return FieldPredicate(
        field,
        lambda v: isinstance(_ensure_scalar_or_raise(v, field), (int, float))
        and v >= threshold,
    )


def where_lt(field: str, threshold: float):
    return FieldPredicate(
        field,
        lambda v: isinstance(_ensure_scalar_or_raise(v, field), (int, float))
        and v < threshold,
    )


def where_lte(field: str, threshold: float):
    return FieldPredicate(
        field,
        lambda v: isinstance(_ensure_scalar_or_raise(v, field), (int, float))
        and v <= threshold,
    )


# === Combinators ===
def and_(*preds):
    return AndPredicate(*preds)


def or_(*preds):
    return OrPredicate(*preds)


def not_(pred):
    return NotPredicate(pred)
