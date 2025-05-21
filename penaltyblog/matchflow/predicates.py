from typing import Any, Callable

from .steps.utils import get_field


# === Base Predicate ===
class Predicate:
    def __call__(self, record: dict) -> bool:
        raise NotImplementedError

    def __and__(self, other):
        return AndPredicate(self, other)

    def __or__(self, other):
        return OrPredicate(self, other)

    def __invert__(self):
        return NotPredicate(self)


# === Core Predicate Types ===
class FieldPredicate(Predicate):
    def __init__(self, field: str, fn: Callable[[Any], bool]):
        self.field = field
        self.fn = fn

    def __call__(self, record: dict) -> bool:
        value = get_field(record, self.field)
        return self.fn(value)

    def __repr__(self):
        return f"<FieldPredicate: {self.field}>"


class NotPredicate(Predicate):
    def __init__(self, pred: Predicate):
        self.pred = pred

    def __call__(self, record: dict) -> bool:
        return not self.pred(record)

    def __repr__(self):
        return f"<Not({self.pred})>"


class AndPredicate(Predicate):
    def __init__(self, *preds: Predicate):
        self.preds = preds

    def __call__(self, record: dict) -> bool:
        return all(pred(record) for pred in self.preds)

    def __repr__(self):
        return f"<And({', '.join(map(str, self.preds))})>"


class OrPredicate(Predicate):
    def __init__(self, *preds: Predicate):
        self.preds = preds

    def __call__(self, record: dict) -> bool:
        return any(pred(record) for pred in self.preds)

    def __repr__(self):
        return f"<Or({', '.join(map(str, self.preds))})>"
