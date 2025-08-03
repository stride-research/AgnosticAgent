from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_ai.llm import (
    AgentLLM,
    LLMConfig,
    LLMMessage,
    LLMProvider,
    LLMResponse,
    OpenRouterProvider,
    UnifiedLLMClient,
)


class TestLLMInterface:
    """Test core LLM interface functionality."""

    def test_llm_message_creation(self):
        """Test LLMMessage model creation."""
        message = LLMMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.name is None

    def test_llm_config_creation(self):
        """Test LLMConfig model creation."""
        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="gpt-3.5-turbo",
            temperature=0.5
        )
        assert config.provider == LLMProvider.OPENROUTER
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.5

    def test_llm_response_creation(self):
        """Test LLMResponse model creation."""
        response = LLMResponse(
            content="Hello there!",
            model="gpt-3.5-turbo",
            provider=LLMProvider.OPENROUTER
        )
        assert response.content == "Hello there!"
        assert response.model == "gpt-3.5-turbo"
        assert response.provider == LLMProvider.OPENROUTER


class TestUnifiedLLMClient:
    """Test UnifiedLLMClient functionality."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = MagicMock()
        provider.config.provider = LLMProvider.OPENROUTER
        provider.complete = AsyncMock(return_value=LLMResponse(
            content="Test response",
            model="test-model",
            provider=LLMProvider.OPENROUTER
        ))
        return provider

    def test_provider_registration(self, mock_provider):
        """Test provider registration."""
        client = UnifiedLLMClient()
        client.register_provider(mock_provider)

        providers = client.list_providers()
        assert LLMProvider.OPENROUTER in providers

    @pytest.mark.asyncio
    async def test_completion_routing(self, mock_provider):
        """Test completion routing to correct provider."""
        client = UnifiedLLMClient()
        client.register_provider(mock_provider)

        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="test-model"
        )
        messages = [LLMMessage(role="user", content="Hello")]

        response = await client.complete(messages, config)

        assert response.content == "Test response"
        mock_provider.complete.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_unregistered_provider_error(self):
        """Test error when using unregistered provider."""
        client = UnifiedLLMClient()
        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="test-model"
        )
        messages = [LLMMessage(role="user", content="Hello")]

        with pytest.raises(ValueError, match="Provider openrouter not registered"):
            await client.complete(messages, config)


class TestAgentLLM:
    """Test AgentLLM simplified interface."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock unified client."""
        client = MagicMock()
        client.complete = AsyncMock(return_value=LLMResponse(
            content="Agent response",
            model="test-model",
            provider=LLMProvider.OPENROUTER
        ))
        return client

    @pytest.mark.asyncio
    async def test_think_method(self, mock_client):
        """Test the think method."""
        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="test-model"
        )
        agent_llm = AgentLLM(mock_client, config)

        response = await agent_llm.think("What is 2+2?")

        assert response == "Agent response"
        mock_client.complete.assert_called_once()

        # Check that the correct message was sent
        call_args = mock_client.complete.call_args
        messages = call_args[0][0]
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "What is 2+2?"

    @pytest.mark.asyncio
    async def test_think_with_context(self, mock_client):
        """Test the think method with context."""
        config = LLMConfig(
            provider=LLMProvider.OPENROUTER,
            model="test-model"
        )
        agent_llm = AgentLLM(mock_client, config)

        await agent_llm.think("What is 2+2?", context="You are a math tutor")

        call_args = mock_client.complete.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[0].content == "You are a math tutor"
        assert messages[1].role == "user"
        assert messages[1].content == "What is 2+2?"