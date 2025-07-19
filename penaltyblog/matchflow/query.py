import ast
import re
from typing import Any, Callable, Dict

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


def parse_query_expr(expr: str, local_vars: dict[str, Any]) -> Callable:
    """
    Parses a query string with @var references into a predicate.

    Args:
        expr (str): The query string to parse.
        local_vars (dict[str, Any]): The local variables to use for variable resolution.
    Returns:
        Callable: The parsed predicate.
    Raises:
        ValueError: If the expression is invalid.
    """
    # Replace @foo with __query_var_foo to make it valid Python
    rewritten = re.sub(r"@([a-zA-Z_][a-zA-Z0-9_]*)", r"__query_var_\1", expr)

    # 2. Parse safely with ast
    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid query syntax: {e}")

    # 3. Rewrite local_vars to match __query_var_* keys
    scoped_vars = {f"__query_var_{k}": v for k, v in local_vars.items()}

    # 4. Convert to predicate
    return _convert_ast(tree.body, local_vars=scoped_vars)


def _convert_ast(node, local_vars: Dict[str, Any] = None):
    """
    Convert an AST node into a Predicate.

    Args:
        node (ast.AST): The AST node to convert.
    Returns:
        Predicate: The converted Predicate.
    Raises:
        ValueError: If the node is not a boolean operation or comparison.
    """
    local_vars = local_vars or {}
    if isinstance(node, ast.BoolOp):
        op = node.op
        values = [_convert_ast(v, local_vars) for v in node.values]
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
        value = _eval_literal(right, local_vars)

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
        args = [_eval_literal(arg, local_vars) for arg in node.args]

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


def _eval_literal(node, local_vars=None):
    """
    Evaluate a literal AST node, possibly resolving variables from `local_vars`.

    Args:
        node (ast.AST): The AST node to evaluate.
        local_vars (dict, optional): A mapping of variable names to values (used for @var support).

    Returns:
        Any: The evaluated value.

    Raises:
        ValueError: If the node is unsupported or a variable is undefined.
    """
    local_vars = local_vars or {}

    if isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(
        node, ast.Name
    ):  # Variable (e.g. @threshold → __query_var_threshold)
        if node.id in local_vars:
            return local_vars[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    elif isinstance(node, ast.List):
        return [_eval_literal(elt, local_vars) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_literal(elt, local_vars) for elt in node.elts)
    elif hasattr(ast, "Str") and isinstance(node, ast.Str):  # Legacy
        return node.s
    elif hasattr(ast, "Num") and isinstance(node, ast.Num):  # Legacy
        return node.n
    else:
        raise ValueError(f"Unsupported literal: {ast.dump(node)}")
