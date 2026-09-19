import asyncio

from app.llm.base import ToolDefinition
from app.tools.builtin_tools import CALCULATOR_DEFINITION, calculator, get_current_datetime
from app.tools.registry import ToolRegistry


def test_calculator_basic_arithmetic():
    assert asyncio.run(calculator("2 + 2")) == "4"
    assert asyncio.run(calculator("(3 + 4) * 2")) == "14"
    assert asyncio.run(calculator("10 / 4")) == "2.5"
    assert asyncio.run(calculator("2 ** 10")) == "1024"


def test_calculator_rejects_unsafe_input_instead_of_executing_it():
    # No eval() under the hood: anything that isn't pure arithmetic on
    # literals (names, calls, attribute access, imports, ...) must fail
    # safely rather than execute.
    for expr in ["__import__('os').system('echo pwned')", "open('/etc/passwd')", "[].__class__", "1 and 2"]:
        result = asyncio.run(calculator(expr))
        assert result.startswith("Could not evaluate")


def test_calculator_definition_matches_strict_schema_requirements():
    # Strict tool use (Anthropic/OpenAI) requires additionalProperties: False
    # and every property listed as required.
    schema = CALCULATOR_DEFINITION.parameters
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"].keys())


def test_get_current_datetime_returns_iso8601_utc():
    result = asyncio.run(get_current_datetime())
    assert result.endswith("+00:00")
    # Should parse back as a valid ISO-8601 timestamp.
    from datetime import datetime

    datetime.fromisoformat(result)


def test_registry_dispatches_to_registered_handler():
    registry = ToolRegistry()
    registry.register(CALCULATOR_DEFINITION, calculator)

    result, is_error = asyncio.run(registry.call("calculator", {"expression": "6 * 7"}))
    assert result == "42"
    assert is_error is False


def test_registry_reports_unknown_tool_as_error_without_raising():
    registry = ToolRegistry()
    result, is_error = asyncio.run(registry.call("does_not_exist", {}))
    assert is_error is True
    assert "unknown tool" in result.lower()


def test_registry_reports_bad_arguments_as_error_without_raising():
    registry = ToolRegistry()
    registry.register(CALCULATOR_DEFINITION, calculator)
    result, is_error = asyncio.run(registry.call("calculator", {"wrong_arg": "1+1"}))
    assert is_error is True


def test_registry_definitions_reflect_registered_tools():
    registry = ToolRegistry()
    registry.register(CALCULATOR_DEFINITION, calculator)
    definitions = registry.definitions()
    assert len(definitions) == 1
    assert isinstance(definitions[0], ToolDefinition)
    assert definitions[0].name == "calculator"
