from agentforge.agents.architectures.react.schemas import (
    ReActDecision,
    ToolCall,
)
from agentforge.agents.architectures.react.state import ReActState


def test_default_state():
    state = ReActState()

    assert state.messages == []
    assert state.iteration == 0
    assert state.action is None
    assert state.observation is None
    assert state.finished is False


def test_state_accepts_messages():
    messages = [
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    state = ReActState(messages=messages)

    assert state.messages == messages


def test_state_accepts_iteration():
    state = ReActState(iteration=3)

    assert state.iteration == 3


def test_state_accepts_tool_call_action():
    action = ReActDecision(
        type="tool_call",
        tool_call=ToolCall(
            tool_name="calculator",
            arguments={"expression": "2 + 2"},
        ),
    )

    state = ReActState(action=action)

    assert state.action == action
    assert state.action.type == "tool_call"


def test_state_accepts_observation():
    observation = {"result": 4}

    state = ReActState(observation=observation)

    assert state.observation == observation


def test_state_can_be_finished():
    state = ReActState(finished=True)

    assert state.finished is True


def test_default_messages_are_not_shared():
    first = ReActState()
    second = ReActState()

    first.messages.append(
        {"role": "user", "content": "Hello"}
    )

    assert second.messages == []
