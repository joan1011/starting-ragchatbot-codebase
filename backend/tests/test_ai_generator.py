"""
Tests for AIGenerator in ai_generator.py.

Covers:
- Direct (non-tool) response path
- Tool-use two-turn flow: tool called, result injected, final response returned
- Conversation history injected into system prompt
- Tool definitions included in api params when tools present
- Tool result correctly appended to messages
- API model ID validity (detects wrong model name causing all queries to fail)
"""
import pytest
from unittest.mock import MagicMock, patch, call
from types import SimpleNamespace

# ── Path setup is handled by conftest.py ─────────────────────────────────────
from ai_generator import AIGenerator


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_content_block(type_="text", text="Hello!", tool_name=None, tool_id=None, tool_input=None):
    """Build a fake Anthropic content block."""
    block = MagicMock()
    block.type = type_
    if type_ == "text":
        block.text = text
    if type_ == "tool_use":
        block.name = tool_name
        block.id = tool_id
        block.input = tool_input or {}
    return block


def _make_response(stop_reason="end_turn", text="Answer here", content_blocks=None):
    """Build a fake Anthropic messages response."""
    resp = MagicMock()
    resp.stop_reason = stop_reason
    if content_blocks is None:
        resp.content = [_make_content_block(type_="text", text=text)]
    else:
        resp.content = content_blocks
    return resp


def _make_generator(model="claude-sonnet-4-20250514"):
    return AIGenerator(api_key="test-key", model=model)


# ── Tests: direct response path ──────────────────────────────────────────────

class TestDirectResponse:

    def test_returns_text_from_end_turn_response(self):
        """When stop_reason is end_turn, return the text content directly."""
        gen = _make_generator()
        direct_resp = _make_response(stop_reason="end_turn", text="Direct answer")

        with patch.object(gen.client.messages, "create", return_value=direct_resp):
            result = gen.generate_response(query="What is X?")

        assert result == "Direct answer"

    def test_does_not_call_tool_manager_on_end_turn(self):
        """Tool manager should NOT be called when stop_reason is end_turn."""
        gen = _make_generator()
        direct_resp = _make_response(stop_reason="end_turn")
        tool_manager = MagicMock()

        with patch.object(gen.client.messages, "create", return_value=direct_resp):
            gen.generate_response(query="Hi", tool_manager=tool_manager)

        tool_manager.execute_tool.assert_not_called()

    def test_tools_added_to_api_params_when_provided(self):
        """Tools and tool_choice should appear in the API call when tools are given."""
        gen = _make_generator()
        direct_resp = _make_response()
        tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]

        with patch.object(gen.client.messages, "create", return_value=direct_resp) as mock_create:
            gen.generate_response(query="q", tools=tools)
            call_kwargs = mock_create.call_args[1]

        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    def test_no_tools_key_when_tools_not_provided(self):
        """If no tools argument, 'tools' should not appear in the API params."""
        gen = _make_generator()
        direct_resp = _make_response()

        with patch.object(gen.client.messages, "create", return_value=direct_resp) as mock_create:
            gen.generate_response(query="q")
            call_kwargs = mock_create.call_args[1]

        assert "tools" not in call_kwargs

    def test_conversation_history_appended_to_system_prompt(self):
        """History should be included in the system parameter, not messages."""
        gen = _make_generator()
        direct_resp = _make_response()
        history = "User: hello\nAssistant: hi"

        with patch.object(gen.client.messages, "create", return_value=direct_resp) as mock_create:
            gen.generate_response(query="What next?", conversation_history=history)
            call_kwargs = mock_create.call_args[1]

        assert "Previous conversation" in call_kwargs["system"]
        assert "hello" in call_kwargs["system"]


# ── Tests: tool-use two-turn flow ─────────────────────────────────────────────

class TestToolUseTwoTurnFlow:

    def _setup_tool_use_flow(self, tool_result_text="[MCP Course - Lesson 1]\nsome content"):
        """Return (generator, mock_create, tool_manager) ready for a tool-use test."""
        gen = _make_generator()
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = tool_result_text

        # First API call returns tool_use
        tool_block = _make_content_block(
            type_="tool_use",
            tool_name="search_course_content",
            tool_id="tu_001",
            tool_input={"query": "MCP basics"},
        )
        first_resp = _make_response(stop_reason="tool_use", content_blocks=[tool_block])

        # Second API call returns final answer
        second_resp = _make_response(stop_reason="end_turn", text="Final synthesized answer")

        mock_create = MagicMock(side_effect=[first_resp, second_resp])
        gen.client.messages.create = mock_create

        return gen, mock_create, tool_manager

    def test_tool_manager_called_with_correct_tool_name_and_args(self):
        gen, mock_create, tool_manager = self._setup_tool_use_flow()

        gen.generate_response(
            query="Explain MCP", tools=[{}], tool_manager=tool_manager
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="MCP basics"
        )

    def test_final_response_text_returned(self):
        """generate_response() should return the text from the second API call."""
        gen, mock_create, tool_manager = self._setup_tool_use_flow()

        result = gen.generate_response(
            query="Explain MCP", tools=[{}], tool_manager=tool_manager
        )

        assert result == "Final synthesized answer"

    def test_two_api_calls_made(self):
        """Exactly two calls to messages.create for the tool-use flow."""
        gen, mock_create, tool_manager = self._setup_tool_use_flow()

        gen.generate_response(query="Explain MCP", tools=[{}], tool_manager=tool_manager)

        assert mock_create.call_count == 2

    def test_tool_result_injected_into_second_call_messages(self):
        """Second API call must include a 'tool_result' message with the tool output."""
        gen, mock_create, tool_manager = self._setup_tool_use_flow(
            tool_result_text="search results here"
        )

        gen.generate_response(query="Explain MCP", tools=[{}], tool_manager=tool_manager)

        second_call_kwargs = mock_create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Find the user message (role=="user") whose content is a list of dicts —
        # i.e. the tool_result carrier. Exclude the first user message (plain string)
        # and the assistant message (list of SDK mock objects).
        tool_result_messages = [
            m for m in messages
            if m["role"] == "user"
            and isinstance(m["content"], list)
            and len(m["content"]) > 0
            and isinstance(m["content"][0], dict)
        ]
        assert tool_result_messages, (
            "No user-role tool_result message found in second API call messages. "
            f"Messages: {[{'role': m['role'], 'content_type': type(m['content']).__name__} for m in messages]}"
        )

        tool_result_block = tool_result_messages[0]["content"][0]
        assert tool_result_block["type"] == "tool_result"
        assert tool_result_block["content"] == "search results here"
        assert tool_result_block["tool_use_id"] == "tu_001"

    def test_second_api_call_has_no_tools_key(self):
        """The second (synthesis) API call should NOT include tools."""
        gen, mock_create, tool_manager = self._setup_tool_use_flow()

        gen.generate_response(query="Explain MCP", tools=[{}], tool_manager=tool_manager)

        second_call_kwargs = mock_create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs

    def test_assistant_tool_use_block_in_second_call_messages(self):
        """The assistant's tool_use block must appear before the tool_result."""
        gen, mock_create, tool_manager = self._setup_tool_use_flow()

        gen.generate_response(query="Explain MCP", tools=[{}], tool_manager=tool_manager)

        second_call_kwargs = mock_create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        roles = [m["role"] for m in messages]

        # Should be: user (original query), assistant (tool_use), user (tool_result)
        assert roles.count("assistant") >= 1
        assistant_idx = next(i for i, r in enumerate(roles) if r == "assistant")
        user_after_assistant = any(r == "user" for r in roles[assistant_idx + 1:])
        assert user_after_assistant, "tool_result user message must follow assistant tool_use"


# ── Tests: model ID validity ──────────────────────────────────────────────────

class TestModelIdValidity:
    """
    The model string in config.py is 'claude-sonnet-4-20250514'.
    This test verifies what happens when the Anthropic API is called with this ID.
    It uses a mock to catch the model name passed to the API and flags if it
    looks wrong compared to the known-valid model ID patterns.
    """

    KNOWN_VALID_PREFIXES = [
        "claude-opus-4",
        "claude-sonnet-4",
        "claude-haiku-4",
        "claude-3-5",
        "claude-3-7",
    ]

    def test_configured_model_matches_known_pattern(self):
        """
        Model ID must start with a known-valid prefix.
        If this fails, the API will reject every request with a 'model not found' error,
        causing 'query failed' for all content questions.
        """
        from config import config
        model = config.ANTHROPIC_MODEL
        assert any(model.startswith(p) for p in self.KNOWN_VALID_PREFIXES), (
            f"Model ID '{model}' does not match any known-valid prefix. "
            f"Valid prefixes: {self.KNOWN_VALID_PREFIXES}. "
            "This will cause the Anthropic API to reject ALL requests."
        )

    def test_model_id_passed_to_api_matches_config(self):
        """The model string in base_params must equal config.ANTHROPIC_MODEL."""
        from config import config
        gen = AIGenerator(api_key="test", model=config.ANTHROPIC_MODEL)
        assert gen.base_params["model"] == config.ANTHROPIC_MODEL

    def test_api_error_on_invalid_model_propagates_as_exception(self):
        """
        If the Anthropic API raises an error (e.g. invalid model ID),
        generate_response() should let it propagate so the caller can handle it.
        """
        gen = _make_generator(model="claude-nonexistent-model")
        with patch.object(
            gen.client.messages,
            "create",
            side_effect=Exception("model: claude-nonexistent-model not found"),
        ):
            with pytest.raises(Exception, match="not found"):
                gen.generate_response(query="Hello")
