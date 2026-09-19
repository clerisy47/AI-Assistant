"""Tests the tool-calling loop's mechanics in isolation, using a scripted
fake LLMProvider instead of a real API. This verifies the orchestrator's
control flow (call model -> execute tool -> feed result back -> repeat)
without needing an API key, network access, or a live model."""

import asyncio

from app.agent.orchestrator import AgentOrchestrator
from app.llm.base import LLMMessage, LLMProvider, LLMResponse, ToolCall, ToolDefinition
from app.tools.registry import ToolRegistry


class ScriptedProvider(LLMProvider):
    """Returns pre-scripted responses in order, one per call to `generate`."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[list[LLMMessage]] = []

    async def generate(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(list(messages))
        return self._responses.pop(0)

    async def generate_structured(self, *args, **kwargs):
        raise NotImplementedError("not used in these tests")


async def _echo_tool(value: str) -> str:
    return f"echo:{value}"


def _registry_with_echo_tool() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echoes the given value back.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        _echo_tool,
    )
    return registry


def test_returns_immediately_when_model_makes_no_tool_call():
    provider = ScriptedProvider([LLMResponse(content="Hello!", tool_calls=[], stop_reason="end_turn")])
    orchestrator = AgentOrchestrator(provider, ToolRegistry())

    result = asyncio.run(orchestrator.run([LLMMessage(role="user", content="hi")]))

    assert result["answer"] == "Hello!"
    assert result["tool_trace"] == []
    assert result["iterations"] == 1
    assert len(provider.calls) == 1


def test_executes_tool_call_and_feeds_result_back_to_the_model():
    provider = ScriptedProvider(
        [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="call_1", name="echo", arguments={"value": "hi"})],
                stop_reason="tool_use",
            ),
            LLMResponse(content="The tool said echo:hi", tool_calls=[], stop_reason="end_turn"),
        ]
    )
    orchestrator = AgentOrchestrator(provider, _registry_with_echo_tool())

    result = asyncio.run(orchestrator.run([LLMMessage(role="user", content="please echo hi")]))

    assert result["answer"] == "The tool said echo:hi"
    assert result["iterations"] == 2
    assert len(result["tool_trace"]) == 1
    assert result["tool_trace"][0] == {"tool": "echo", "arguments": {"value": "hi"}, "result": "echo:hi", "is_error": False}

    # The second call to the model must include the tool's result as context.
    second_call_messages = provider.calls[1]
    assert any(m.role == "tool" and m.content == "echo:hi" for m in second_call_messages)


def test_stops_after_max_tool_iterations_instead_of_looping_forever():
    always_calls_tool = LLMResponse(
        content=None, tool_calls=[ToolCall(id="x", name="echo", arguments={"value": "loop"})], stop_reason="tool_use"
    )
    provider = ScriptedProvider([always_calls_tool] * 10)
    orchestrator = AgentOrchestrator(provider, _registry_with_echo_tool(), max_tool_iterations=3)

    result = asyncio.run(orchestrator.run([LLMMessage(role="user", content="loop forever")]))

    assert result["iterations"] == 3
    assert len(result["tool_trace"]) == 3
    assert len(provider.calls) == 3  # never attempted a 4th round-trip
    assert "final answer" in result["answer"].lower()


def test_unknown_tool_is_reported_as_an_error_without_crashing():
    provider = ScriptedProvider(
        [
            LLMResponse(content=None, tool_calls=[ToolCall(id="x", name="does_not_exist", arguments={})], stop_reason="tool_use"),
            LLMResponse(content="done", tool_calls=[], stop_reason="end_turn"),
        ]
    )
    orchestrator = AgentOrchestrator(provider, ToolRegistry())  # empty registry

    result = asyncio.run(orchestrator.run([LLMMessage(role="user", content="call a missing tool")]))

    assert result["tool_trace"][0]["is_error"] is True
    assert result["answer"] == "done"  # the loop recovers instead of crashing
