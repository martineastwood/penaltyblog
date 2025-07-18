import ast
import warnings

from .predicates_helpers import (
    and_,
    not_,
    or_,
    where_contains,
    where_equals,
    where_exists,
    where_gt,
    where_gte,
    where_in,
    where_is_null,
    where_lt,
    where_lte,
    where_not_equals,
    where_not_in,
)


def parse_query_expr(expr: str):
    """
    Parse a simple boolean expression like "age > 30 and city == 'London'"
    into a Predicate.

    Args:
        expr (str): The boolean expression to parse.
    Returns:
        Predicate: The parsed Predicate.
    Raises:
        ValueError: If the expression is invalid.
    """
    try:
        tree = ast.parse(expr, mode="eval")
        return _convert_ast(tree.body)
    except SyntaxError as e:
        raise ValueError(f"Invalid query syntax: {e}")


def _convert_ast(node):
    """
    Convert an AST node into a Predicate.

    Args:
        node (ast.AST): The AST node to convert.
    Returns:
        Predicate: The converted Predicate.
    Raises:
        ValueError: If the node is not a boolean operation or comparison.
    """
    if isinstance(node, ast.BoolOp):
        op = node.op
        values = [_convert_ast(v) for v in node.values]
        if isinstance(op, ast.And):
            return and_(*values)
        elif isinstance(op, ast.Or):
            return or_(*values)
        else:
            raise ValueError(f"Unsupported boolean op: {ast.dump(op)}")

    elif isinstance(node, ast.Compare):
        left = node.left
        field = _extract_field(left)
        ops = node.ops
        comparators = node.comparators

        if len(ops) > 1:
            raise ValueError("Chained comparisons not supported")

        op = ops[0]
        right = comparators[0]

        # Special case: field is None / is not None
        if isinstance(op, ast.Is):
            if isinstance(right, ast.Constant) and right.value is None:
                return where_is_null(field)
            else:
                raise ValueError("Only 'is None' comparisons are supported")
        elif isinstance(op, ast.IsNot):
            if isinstance(right, ast.Constant) and right.value is None:
                return where_exists(field)
            else:
                raise ValueError("Only 'is not None' comparisons are supported")

        # Evaluate right-hand literal
        value = _eval_literal(right)

        if isinstance(op, ast.Eq):
            return where_equals(field, value)
        elif isinstance(op, ast.NotEq):
            return where_not_equals(field, value)
        elif isinstance(op, ast.Gt):
            return where_gt(field, value)
        elif isinstance(op, ast.GtE):
            return where_gte(field, value)
        elif isinstance(op, ast.Lt):
            return where_lt(field, value)
        elif isinstance(op, ast.LtE):
            return where_lte(field, value)
        elif isinstance(op, ast.In):
            return where_in(field, value)
        elif isinstance(op, ast.NotIn):
            return where_not_in(field, value)
        else:
            raise ValueError(f"Unsupported comparison: {ast.dump(op)}")

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not_(_convert_ast(node.operand))

    elif isinstance(node, ast.Call):
        field = _extract_field(node.func.value)
        method = node.func.attr
        args = [_eval_literal(arg) for arg in node.args]

        if method == "contains":
            return where_contains(field, *args)
        else:
            raise ValueError(f"Unsupported method call: {method}")

    else:
        raise ValueError(f"Unsupported query expression: {ast.dump(node)}")


def _extract_field(node):
    """
    Extract a field name from an AST node.

    Args:
        node (ast.AST): The AST node to extract the field name from.
    Returns:
        str: The field name.
    Raises:
        ValueError: If the node is not a name or attribute.
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            return ".".join(reversed(parts))
    raise ValueError(f"Unsupported field expression: {ast.dump(node)}")


def _eval_literal(node):
    """
    Evaluate a literal AST node.

    Args:
        node (ast.AST): The AST node to evaluate.
    Returns:
        Any: The evaluated value.
    Raises:
        ValueError: If the node is not a constant, list, or tuple.
    """
    if isinstance(node, ast.Constant):
        return node.value
    # Handle ast.List
    elif isinstance(node, ast.List):
        return [_eval_literal(elt) for elt in node.elts]
    # Handle ast.Tuple
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_literal(elt) for elt in node.elts)
    elif hasattr(ast, "Str") and isinstance(node, ast.Str):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return node.s
    elif hasattr(ast, "Num") and isinstance(node, ast.Num):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return node.n
    else:
        raise ValueError(f"Unsupported literal: {ast.dump(node)}")
