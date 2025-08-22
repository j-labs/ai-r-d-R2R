import logging
import os
from collections.abc import Generator, AsyncGenerator
from typing import Any

from openai import AsyncOpenAI, OpenAI

from core.base.abstractions import GenerationConfig, LLMChatCompletionChunk
from core.base.providers.llm import CompletionConfig, CompletionProvider

logger = logging.getLogger()


class VLLMCompletionProvider(CompletionProvider):
    """
    A minimal provider for VLLM LLM inference endpoints.
    Allows choosing between fast and normal LLM at runtime via GenerationConfig.
    """

    def __init__(self, config: CompletionConfig, *args, **kwargs) -> None:
        super().__init__(config)

        self.normal_llm_base_url = os.getenv("VLLM_QUALITY_LLM")
        logger.info(f"VLLM_QUALITY_LLM environment variable set to {self.normal_llm_base_url}")
        if self.normal_llm_base_url:
            self.normal_llm_client = OpenAI(
                base_url=f"{self.normal_llm_base_url}/v1",
                api_key="dummy",
            )
            self.async_normal_llm_client = AsyncOpenAI(
                base_url=f"{self.normal_llm_base_url}/v1",
                api_key="dummy",
            )
            logger.info("Quality LLM clients initialized successfully")
        else:
            self.normal_llm_client = None
            self.async_normal_llm_client = None
            logger.warning("VLLM_QUALITY_LLM environment variable not set")

        self.fast_llm_base_url = os.getenv("HUGGINGFACE_FAST_LLM")
        logger.info(f"VLLM_FAST_LLM environment variable set to {self.fast_llm_base_url}")
        if self.fast_llm_base_url:
            self.fast_llm_client = OpenAI(
                api_key="dummy",  # Not used but required
                base_url=f"{self.fast_llm_base_url}/v1",
            )
            self.async_fast_llm_client = AsyncOpenAI(
                api_key="dummy",  # Not used but required
                base_url=f"{self.fast_llm_base_url}/v1",
            )
            logger.info("Fast LLM clients initialized successfully")
        else:
            self.fast_llm_client = None
            self.async_fast_llm_client = None
            logger.warning("VLLM_FAST_LLM environment variable not set")

        if not any([self.normal_llm_client, self.fast_llm_client]):
            raise ValueError(
                "No valid client credentials found. Please set either VLLM_QUALITY_LLM or VLLM_FAST_LLM environment variables."
            )

    def _get_client_and_model(self, model: str) -> tuple[OpenAI, str]:
        """
        Determine which client to use based on the model parameter in generation_config.
        If model is "fast", use the fast LLM client, otherwise use the normal LLM client.
        """
        use_fast_llm = False

        if "/fast/" in model:
            use_fast_llm = True

        if use_fast_llm:
            if not self.fast_llm_client:
                raise ValueError(
                    "Fast LLM client not configured but requested to use fast LLM"
                )
            return self.fast_llm_client, model.split("/")[-1]
        else:
            if not self.normal_llm_client:
                raise ValueError(
                    "Quality LLM client not configured and no fallback available"
                )
            return self.normal_llm_client, model.split("/")[-1]

    def _get_async_client_and_model(self, model: str) -> tuple[AsyncOpenAI, str]:
        """
        Determine which async client to use based on the model parameter in generation_config.
        If model is "fast", use the fast LLM client, otherwise use the normal LLM client.
        """
        use_fast_llm = False

        if "/fast/" in model:
            use_fast_llm = True
            logger.info(f"Using fast LLM HF client for async task with model {model}")

        if use_fast_llm:
            if not self.async_fast_llm_client:
                raise ValueError(
                    "Fast LLM client not configured but requested to use fast LLM"
                )
            return self.async_fast_llm_client, model.split("/")[-1]
        else:
            if not self.async_normal_llm_client:
                raise ValueError(
                    "Quality LLM client not configured and no fallback available"
                )
            return self.async_normal_llm_client, model.split("/")[-1]

    @staticmethod
    def _get_base_args(generation_config: GenerationConfig):
        """Extract arguments from GenerationConfig."""
        args: dict[str, Any] = {
            "model": "/".join(generation_config.model.split("/")[-2:]),
            "stream": generation_config.stream,
            "tools": generation_config.tools,
        }

        if isinstance(generation_config.top_p, (int, float)):
            args["top_p"] = generation_config.top_p - 0.01 if generation_config.top_p > 0.01 else generation_config.top_p + 0.01

        args["max_tokens"] = max(generation_config.max_tokens_to_sample, 4096)
        args["temperature"] = generation_config.temperature

        # Add any additional kwargs
        if generation_config.add_generation_kwargs:
            args.update(generation_config.add_generation_kwargs)

        return args

    async def _execute_task(self, task: dict[str, Any]):
        """Execute a completion task asynchronously."""
        messages = task["messages"]
        generation_config = task["generation_config"]
        logger.info(f"Generation config: {generation_config}; messages: {messages}")
        kwargs = task["kwargs"]

        args = self._get_base_args(generation_config)
        client, _ = self._get_async_client_and_model(generation_config.model)
        args["messages"] = messages
        args = {**args, **kwargs}

        logger.info(f"Executing async task with args: {args}")
        try:
            if args.get("stream", False):
                # For streaming, return the generator directly
                return self._acompletion_stream(args, client)
            else:
                response = await client.chat.completions.create(**args)
            logger.debug("Async task executed successfully")
            return response
        except Exception as e:
            logger.error(f"Async task execution failed: {str(e)}")
            raise

    @staticmethod
    async def _acompletion_stream(
            args: dict[str, Any],
            client: AsyncOpenAI,
    ) -> AsyncGenerator[LLMChatCompletionChunk, None]:
        """
        Handle streaming completions for VLLM LLM using OpenAI interface.
        """
        chunk_count = 0

        try:
            stream = await client.chat.completions.create(**args)
            async for chunk in stream:
                chunk_count += 1
                try:
                    # OpenAI chunks are already in the format we need, just convert to our type
                    chunk_dict = chunk.model_dump()
                    yield LLMChatCompletionChunk(**chunk_dict)
                    logger.debug(f"Yielding chunk #{chunk_count}: {chunk_dict}")
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk #{chunk_count}: {str(chunk_error)}")
                    # Continue processing other chunks instead of failing completely
                    continue

            logger.info(f"Stream completed. Total chunks: {chunk_count}.")

        except Exception as e:
            logger.error(f"Async VLLM streaming execution failed: {str(e)}")
            raise

    @staticmethod
    def _completion_stream(
        args: dict[str, Any],
        client: OpenAI,
            ) -> Generator[LLMChatCompletionChunk, None, None]:
        """
        Handle synchronous streaming completions for VLLM LLM using OpenAI interface.
        """
        chunk_count = 0

        try:
            stream = client.chat.completions.create(**args)

            for chunk in stream:
                chunk_count += 1
                try:
                    chunk_dict = chunk.model_dump()
                    yield LLMChatCompletionChunk(**chunk_dict)
                    logger.debug(f"Yielding chunk #{chunk_count}: {chunk_dict}")
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk #{chunk_count}: {str(chunk_error)}")
                    continue

            logger.info(f"Stream completed. Total chunks: {chunk_count}.")

        except Exception as e:
            logger.error(f"Sync VLLM streaming execution failed: {str(e)}")
            raise

    def _execute_task_sync(self, task: dict[str, Any]):
        """Execute a completion task synchronously."""
        messages = task["messages"]
        generation_config = task["generation_config"]
        kwargs = task["kwargs"]

        args = self._get_base_args(generation_config)
        client, _ = self._get_client_and_model(generation_config.model)
        args["messages"] = messages
        args = {**args, **kwargs}

        logger.debug(f"Executing sync task with args: {args}")
        try:
            if args.get("stream", False):
                # For streaming, return the generator directly
                return self._completion_stream(args, client)
            else:
                response = client.chat.completions.create(**args)
                logger.debug("Sync task executed successfully")
                return response
        except Exception as e:
            logger.error(f"Sync task execution failed: {str(e)}")
            raise