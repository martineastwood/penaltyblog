import ast
import re
from datetime import date, datetime
from typing import Any, Callable, Dict, Optional

from .predicates_helpers import (
    and_,
    not_,
    or_,
    where_contains,
    where_endswith,
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
    where_regex_match,
    where_startswith,
)
from .steps.utils import get_field


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
    rewritten = re.sub(r"@([a-zA-Z_][a-zA-Z0-9_]*)", r"__query_var_\1", expr)

    try:
        tree = ast.parse(rewritten, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid query syntax: {e}")

    scoped_vars = {f"__query_var_{k}": v for k, v in local_vars.items()}

    return _convert_ast(tree.body, local_vars=scoped_vars)


def _parse_field_expr(node: ast.AST) -> tuple[str, Callable | None]:
    """
    Parses a node that can be a field or a simple method call on a field.
    Returns (field_name, transform_function).
    Transform function is None if no transformation is applied.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
    ):
        if len(node.args) != 1:
            raise ValueError("len() expects exactly one argument.")
        field = _extract_field(node.args[0])
        return field, len

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        method = node.func.attr
        if method in ("lower", "upper"):
            if node.args or node.keywords:
                raise ValueError(
                    f"Method '{method}' should have no arguments in a query."
                )

            field = _extract_field(node.func.value)
            transform = str.lower if method == "lower" else str.upper
            return field, transform

    # Not a supported method call, assume it's just a field.
    field = _extract_field(node)
    return field, None


def _convert_ast(node, local_vars: Optional[Dict[str, Any]] = None):
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
        # Handle chained comparisons by breaking them into a series of `and`ed comparisons
        if len(node.ops) > 1:
            predicates = []
            all_parts = [node.left] + node.comparators
            for i, op in enumerate(node.ops):
                left_node = all_parts[i]
                right_node = all_parts[i + 1]
                sub_compare_node = ast.Compare(
                    left=left_node, ops=[op], comparators=[right_node]
                )
                predicates.append(_convert_ast(sub_compare_node, local_vars))
            return and_(*predicates)

        # Logic for simple (non-chained) comparisons
        left = node.left
        op = node.ops[0]
        right = node.comparators[0]

        # Special case: field is None / is not None
        if isinstance(op, ast.Is):
            field = _extract_field(left)
            if isinstance(right, ast.Constant) and right.value is None:
                return where_is_null(field)
            raise ValueError("Only 'is None' comparisons are supported")
        if isinstance(op, ast.IsNot):
            field = _extract_field(left)
            if isinstance(right, ast.Constant) and right.value is None:
                return where_exists(field)
            raise ValueError("Only 'is not None' comparisons are supported")

        try:
            field, transform = _parse_field_expr(left)
            value = _eval_literal(right, local_vars)
            op_to_use = op
        except ValueError:
            if isinstance(op, (ast.In, ast.NotIn)):
                raise ValueError(
                    "Field must be on the left for 'in' and 'not in' operators"
                )

            field, transform = _parse_field_expr(right)
            value = _eval_literal(left, local_vars)

            if isinstance(op, ast.Gt):
                op_to_use = ast.Lt()
            elif isinstance(op, ast.GtE):
                op_to_use = ast.LtE()
            elif isinstance(op, ast.Lt):
                op_to_use = ast.Gt()
            elif isinstance(op, ast.LtE):
                op_to_use = ast.GtE()
            else:
                op_to_use = op

        if transform and callable(transform):
            # Capture transform in local scope to ensure it's not None
            transform_func = transform

            def predicate(record: dict) -> bool:
                field_val = get_field(record, field)
                if not hasattr(field_val, "__len__"):
                    return False

                try:
                    transformed_val = transform_func(field_val)
                except TypeError:
                    return False

                if isinstance(op_to_use, ast.Eq):
                    return transformed_val == value
                if isinstance(op_to_use, ast.NotEq):
                    return transformed_val != value
                if isinstance(op_to_use, ast.Gt):
                    return transformed_val > value
                if isinstance(op_to_use, ast.GtE):
                    return transformed_val >= value
                if isinstance(op_to_use, ast.Lt):
                    return transformed_val < value
                if isinstance(op_to_use, ast.LtE):
                    return transformed_val <= value
                if isinstance(op_to_use, ast.In):
                    return transformed_val in value
                if isinstance(op_to_use, ast.NotIn):
                    return transformed_val not in value
                raise ValueError(f"Unsupported comparison: {ast.dump(op_to_use)}")

            return predicate

        if isinstance(op_to_use, ast.Eq):
            return where_equals(field, value)
        if isinstance(op_to_use, ast.NotEq):
            return where_not_equals(field, value)
        if isinstance(op_to_use, ast.Gt):
            return where_gt(field, value)
        if isinstance(op_to_use, ast.GtE):
            return where_gte(field, value)
        if isinstance(op_to_use, ast.Lt):
            return where_lt(field, value)
        if isinstance(op_to_use, ast.LtE):
            return where_lte(field, value)
        if isinstance(op_to_use, ast.In):
            return where_in(field, value)
        if isinstance(op_to_use, ast.NotIn):
            return where_not_in(field, value)

        raise ValueError(f"Unsupported comparison: {ast.dump(op)}")

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not_(_convert_ast(node.operand))

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            field = _extract_field(node.func.value)
            method = node.func.attr
            args = [_eval_literal(arg, local_vars) for arg in node.args]

            if method == "contains":
                return where_contains(field, *args)
            elif method == "startswith":
                return where_startswith(field, *args)
            elif method == "endswith":
                return where_endswith(field, *args)
            elif method == "regex" or method == "match":
                if not args:
                    raise ValueError(
                        f"Method '{method}()' requires at least one argument (pattern)"
                    )
                pattern = args[0]
                flags = args[1] if len(args) > 1 else 0
                return where_regex_match(field, pattern, flags)
            elif method in ("lower", "upper"):
                raise ValueError(
                    f"Method '{method}' cannot be used as a predicate. "
                    f"Use it in a comparison, e.g., 'field.{method}() == \"some_value\"'."
                )
            else:
                raise ValueError(f"Unsupported method call: {method}")
        else:
            raise ValueError(f"Unsupported query expression: {ast.dump(node)}")

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

    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.Name):
        if node.id in local_vars:
            return local_vars[node.id]
        raise ValueError(f"Unknown variable: {node.id}")
    elif isinstance(node, ast.List):
        return [_eval_literal(elt, local_vars) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(_eval_literal(elt, local_vars) for elt in node.elts)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            args = [_eval_literal(arg, local_vars) for arg in node.args]

            if func_name == "datetime":
                return datetime(*args)
            elif func_name == "date":
                return date(*args)
            else:
                raise ValueError(f"Unsupported function call: {func_name}")
        else:
            raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
    elif isinstance(node, ast.Attribute):
        # Handle attribute references like re.IGNORECASE
        if isinstance(node.value, ast.Name) and node.value.id in local_vars:
            obj = local_vars[node.value.id]
            if hasattr(obj, node.attr):
                return getattr(obj, node.attr)
        raise ValueError(f"Unknown attribute: {ast.dump(node)}")
    else:
        raise ValueError(f"Unsupported literal: {ast.dump(node)}")
