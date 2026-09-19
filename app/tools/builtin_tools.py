"""Small, dependency-free tools that don't need any external service.

Kept separate from `knowledge_base_tool.py` because that one needs a
`Retriever` injected (dependency on the RAG stack); these two are pure
functions and need nothing.
"""

from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone

from app.llm.base import ToolDefinition

# --- calculator --------------------------------------------------------------------
#
# Deliberately implemented with `ast.parse` + a small, explicit operator
# whitelist instead of `eval()`. `eval()` on model- or user-supplied text is a
# textbook code-execution vulnerability; this evaluator can only ever reach
# arithmetic on numeric literals, nothing else (no names, no attribute
# access, no calls), so there's no attack surface even though the input
# ultimately comes from an LLM's tool-call arguments.

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


async def calculator(expression: str) -> str:
    """Safely evaluate a basic arithmetic expression."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Could not evaluate '{expression}': {exc}"


CALCULATOR_DEFINITION = ToolDefinition(
    name="calculator",
    description=(
        "Evaluate a basic arithmetic expression (+, -, *, /, //, %, **, parentheses, unary minus). "
        "Use this for any math instead of computing it yourself."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '(3 + 4) * 2 / 7'"},
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
)


# --- current datetime ----------------------------------------------------------------


async def get_current_datetime() -> str:
    """Return the current date and time in UTC, ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


DATETIME_DEFINITION = ToolDefinition(
    name="get_current_datetime",
    description="Get the current date and time in UTC (ISO-8601). Use this for any 'today'/'now'/'current time' question.",
    parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
)
