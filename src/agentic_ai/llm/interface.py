from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENROUTER = "openrouter"
    GOOGLE = "google"
    OLLAMA = "ollama"


class LLMMessage(BaseModel):
    """Standard message format for all providers."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None


class LLMResponse(BaseModel):
    """Standard response format from all providers."""
    content: str
    model: str
    provider: LLMProvider
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Generate a single completion."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available."""
        pass


class UnifiedLLMClient:
    """Unified client for all LLM providers."""

    def __init__(self):
        self._providers: Dict[LLMProvider, BaseLLMProvider] = {}

    def register_provider(self, provider: BaseLLMProvider):
        """Register a new provider."""
        self._providers[provider.config.provider] = provider

    async def complete(
        self,
        messages: List[LLMMessage],
        config: LLMConfig,
        **kwargs
    ) -> LLMResponse:
        """Route completion to appropriate provider."""
        provider = self._providers.get(config.provider)
        if not provider:
            raise ValueError(f"Provider {config.provider} not registered")

        return await provider.complete(messages, **kwargs)

    def list_providers(self) -> List[LLMProvider]:
        """List all registered providers."""
        return list(self._providers.keys())

    async def health_check_all(self) -> Dict[LLMProvider, bool]:
        """Check health of all registered providers."""
        results = {}
        for provider_type, provider in self._providers.items():
            try:
                results[provider_type] = await provider.health_check()
            except Exception:
                results[provider_type] = False
        return results