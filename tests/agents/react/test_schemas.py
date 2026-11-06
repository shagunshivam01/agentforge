from pydantic import ValidationError
import pytest

from agentforge.agents.architectures.react.schemas import ReActDecision, ToolCall

def test_tool_call_defaults_arguments_to_empty_dict():
    tool_call = ToolCall(tool_name="calculator")

    assert tool_call.tool_name == "calculator"
    assert tool_call.arguments == {}


def test_tool_call_accepts_arguments():
    tool_call = ToolCall(
        tool_name="calculator",
        arguments={"expression": "2 + 2"},
    )

    assert tool_call.tool_name == "calculator"
    assert tool_call.arguments == {"expression": "2 + 2"}


def test_tool_call_decision():
    decision = ReActDecision(
        type="tool_call",
        tool_call=ToolCall(
            tool_name="calculator",
            arguments={"expression": "2 + 2"},
        ),
    )

    assert decision.type == "tool_call"
    assert not decision.type == "final"


def test_final_decision():
    decision = ReActDecision(
        type="final",
        final_answer="The answer is 4.",
    )

    assert decision.type == "final"
    assert not decision.type == "tool_call"


def test_tool_call_decision_requires_tool_call():
    with pytest.raises(ValidationError):
        ReActDecision(
            type="tool_call",
        )


def test_final_decision_requires_final_answer():
    with pytest.raises(ValidationError):
        ReActDecision(
            type="final",
        )


def test_tool_call_decision_cannot_have_final_answer():
    with pytest.raises(ValidationError):
        ReActDecision(
            type="tool_call",
            tool_call=ToolCall(tool_name="calculator"),
            final_answer="The answer is 4.",
        )


def test_final_decision_cannot_have_tool_call():
    with pytest.raises(ValidationError):
        ReActDecision(
            type="final",
            final_answer="The answer is 4.",
            tool_call=ToolCall(tool_name="calculator"),
        )


def test_invalid_decision_type():
    with pytest.raises(ValidationError):
        ReActDecision(
            type="something_else",
            final_answer="Hello",
        )
