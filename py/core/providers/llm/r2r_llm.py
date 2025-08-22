import logging
import os
from typing import Any

from core.base.abstractions import GenerationConfig
from core.base.providers.llm import CompletionConfig, CompletionProvider

from .anthropic import AnthropicCompletionProvider
from .azure_foundry import AzureFoundryCompletionProvider
from .litellm import LiteLLMCompletionProvider
from .ollama_chat import OllamaCompletionProvider
from .openai import OpenAICompletionProvider
from .vllm_chat import VLLMCompletionProvider

logger = logging.getLogger(__name__)


class R2RCompletionProvider(CompletionProvider):
    """A provider that routes to the right LLM provider (R2R):

    - If `generation_config.model` starts with "anthropic/", call AnthropicCompletionProvider.
    - If it starts with "azure-foundry/", call AzureFoundryCompletionProvider.
    - If it starts with one of the other OpenAI-like prefixes ("openai/", "azure/", "deepseek/", "ollama/", "lmstudio/")
      or has no prefix (e.g. "gpt-4", "gpt-3.5"), call OpenAICompletionProvider.
    - Otherwise, fallback to LiteLLMCompletionProvider.
    """

    def __init__(self, config: CompletionConfig, *args, **kwargs) -> None:
        """Initialize sub-providers for OpenAI, Anthropic, LiteLLM, and Azure
        Foundry."""
        super().__init__(config)
        self.config = config

        logger.info("Initializing R2RCompletionProvider...")
        self._openai_provider = OpenAICompletionProvider(
            self.config, *args, **kwargs
        )
        self._anthropic_provider = AnthropicCompletionProvider(
            self.config, *args, **kwargs
        )
        self._litellm_provider = LiteLLMCompletionProvider(
            self.config, *args, **kwargs
        )
        self._azure_foundry_provider = AzureFoundryCompletionProvider(
            self.config, *args, **kwargs
        )  # New provider
        self._ollama_provider = OllamaCompletionProvider(
            self.config, *args, **kwargs
        )  # New provider
        if os.getenv("VLLM_QUALITY_LLM") or os.getenv("VLLM_FAST_LLM"):
            self._vllm_povider = VLLMCompletionProvider(
                self.config, *args, **kwargs
            )  # New provider

        logger.debug(
            "R2RCompletionProvider initialized with OpenAI, Anthropic, LiteLLM, and Azure Foundry sub-providers."
        )

    def _choose_subprovider_by_model(
        self, model_name: str, is_streaming: bool = False
    ) -> CompletionProvider:
        """Decide which underlying sub-provider to call based on the model name
        (prefix)."""
        # Route to Anthropic if appropriate.
        if model_name.startswith("anthropic/"):
            return self._anthropic_provider

        # Route to Azure Foundry explicitly.
        if model_name.startswith("azure-foundry/"):
            return self._azure_foundry_provider

        # Route to Ollama explicitly.
        if model_name.startswith("ollama/"):
            return self._ollama_provider

        # Route to Huggingface explicitly.
        if model_name.startswith("vllm/"):
            return self._vllm_povider

        # OpenAI-like prefixes.
        openai_like_prefixes = [
            "openai/",
            "azure/",
            "deepseek/",
            # "ollama/",
            "lmstudio/",
        ]
        if (
            any(
                model_name.startswith(prefix)
                for prefix in openai_like_prefixes
            )
            or "/" not in model_name
        ):
            logger.info(
                ">>> R2RCompletionProvider: model_name determines sub-provider to use: OpenAI."
            )
            return self._openai_provider

        return self._litellm_provider

    async def _execute_task(self, task: dict[str, Any]):
        """Pick the sub-provider based on model name and forward the async
        call."""
        generation_config: GenerationConfig = task["generation_config"]
        model_name = generation_config.model
        sub_provider = self._choose_subprovider_by_model(model_name or "")
        return await sub_provider._execute_task(task)

    def _execute_task_sync(self, task: dict[str, Any]):
        """Pick the sub-provider based on model name and forward the sync
        call."""
        generation_config: GenerationConfig = task["generation_config"]
        model_name = generation_config.model
        sub_provider = self._choose_subprovider_by_model(model_name or "")
        return sub_provider._execute_task_sync(task)
