import json
import logging
import os
import random
from collections.abc import AsyncIterator
from typing import Any, AsyncGenerator, Generator

from ollama import AsyncClient, ChatResponse, Client, Message, Tool

from core.base.abstractions import GenerationConfig, LLMChatCompletionChunk
from core.base.providers.llm import CompletionConfig, CompletionProvider
from shared import LLMChatCompletion

logger = logging.getLogger()


class MessageConverter:
    """Converts between R2R message format and Ollama message format"""

    @staticmethod
    def r2r_to_ollama(messages: list[dict]) -> list[Message]:
        """Convert R2R messages to Ollama format"""

        formatted_messages = []

        # Add default system message if none exists
        has_system_message = any(msg["role"] == "system" for msg in messages)
        if not has_system_message:
            logger.info("Adding default system message to Ollama messages.")
            formatted_messages.append(
                Message(
                    role="system",
                    content="You are a helpful assistant. When tools are available, always use them when appropriate to help answer the user's question.",
                )
            )

        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")

            if role == "system":
                formatted_messages.append(
                    Message(role="system", content=content)
                )
            elif role == "user":
                formatted_messages.append(
                    Message(role="user", content=content)
                )
            elif role == "assistant":
                # Handle assistant messages with tool calls
                tool_calls_indicator = msg.get("tool_calls", False)
                tool_calls = []
                if tool_calls_indicator:
                    for tool_call in msg["tool_calls"]:
                        function_data = tool_call.get("function", {})
                        arguments = function_data.get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"Invalid JSON in tool calling arguments: {arguments}"
                                )
                                content += (
                                    f"\n\nEncountered an error while calling tools. "
                                    f"Invalid JSON in tool calling "
                                    f"arguments: {arguments}.\n\n"
                                )
                                arguments = {}

                        tool_calls.append(
                            Message.ToolCall(
                                function=Message.ToolCall.Function(
                                    name=function_data.get("name", ""),
                                    arguments=arguments,
                                )
                            )
                        )
                    formatted_message = Message(
                        role="assistant",
                        content=content,
                        thinking=thinking,
                        tool_calls=tool_calls,
                    )

                    formatted_messages.append(formatted_message)
                else:
                    formatted_messages.append(
                        Message(role="assistant", content=content)
                    )
            else:
                logger.warning(
                    f"Unknown message role: {role}, treating as user message"
                )
                formatted_messages.append(
                    Message(role="user", content=content)
                )

        return formatted_messages

    @staticmethod
    def ollama_tools_from_r2r(tools: list[dict[str, Any]]) -> list[dict]:
        """Convert R2R tools to Ollama's native format"""
        if not tools:
            return []

        ollama_tools = []
        for tool in tools:
            # Create Ollama Tool with Function
            tool_function = Tool.Function(
                name=tool["name"],
                description=tool["function"]["description"],
                parameters=Tool.Function.Parameters(
                    **tool["function"]["parameters"]
                ),
            )

            ollama_tools.append(Tool(function=tool_function).model_dump())

        return ollama_tools

    @staticmethod
    def get_tools_from_message(message: Message) -> list[dict]:
        # Handle both dict and Message object formats
        message_tool_calls = message.tool_calls

        tool_calls = []
        if message_tool_calls:
            for i, tool_call in enumerate(message_tool_calls):
                # Handle both dict and object formats for tool_call
                function_data = tool_call.function
                function_name = (
                    function_data.name
                    if hasattr(function_data, "name")
                    else function_data.get("name", "")
                )
                arguments = (
                    function_data.arguments
                    if hasattr(function_data, "arguments")
                    else function_data.get("arguments", {})
                )

                tool_calls.append(
                    {
                        "id": f"call_{i}",
                        "index": i,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(arguments)
                            if isinstance(arguments, dict)
                            else arguments,
                        },
                    }
                )

        return tool_calls

    def ollama_response_to_r2r(self, response: ChatResponse) -> dict[str, Any]:
        """Format Ollama response to match R2R format for compatibility"""
        logger.debug(f"Raw Ollama response: {response}")

        message = response.message
        content = message.content or ""
        thinking = message.thinking or ""

        # Extract tool calls if present
        tool_calls = self.get_tools_from_message(message)

        # Create a response in OpenAI format
        formatted_response = {
            "id": response.get(
                "id", f"ollama-{response.get('created_at', '')}"
            ),
            "object": "chat.completion",
            "created": int(
                response.get("total_duration", 0) / 1000000
            ),  # Convert ns to s
            "model": response.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls,
                        "thinking": thinking,
                    },
                    "finish_reason": response.get("done_reason", "stop"),
                }
            ],
            "usage": {
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0)
                + response.get("eval_count", 0),
            },
        }

        # Add thinking to the response if present
        if thinking:
            formatted_response["thinking"] = thinking

        return formatted_response

    def ollama_chunk_to_r2r(self, chunk: ChatResponse) -> dict[str, Any]:
        """Format Ollama chunk response to match R2R format for compatibility"""
        logger.debug(f"Raw Ollama chunk response: {chunk}")

        if not chunk:
            logger.warning("Empty chunk received")
            return {}

        # Extract the message content
        message = chunk.message
        content = message.content or ""
        thinking = message.thinking or ""

        # Create a response in OpenAI format
        formatted_chunk = {
            "id": chunk.get(
                "id", f"ollama-chunk-{chunk.get('created_at', '')}"
            ),
            "object": "chat.completion.chunk",
            "created": int(chunk.get("total_duration", 0) / 1000000)
            if chunk.get("total_duration")
            else 0,
            "model": chunk.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content,
                        "thinking": thinking,
                    },
                    "finish_reason": chunk.get("done_reason", "stop")
                    if chunk.get("done", False)
                    else None,
                }
            ],
        }

        # Handle tool calls in streaming chunks
        tool_calls = self.get_tools_from_message(message)
        formatted_chunk["choices"][0]["delta"]["tool_calls"] = tool_calls

        logger.debug(f"Final formatted chunk: {formatted_chunk}")
        return formatted_chunk


class OllamaCompletionProvider(CompletionProvider):
    thinking_models_patterns = ["qwen3"]

    def __init__(self, config: CompletionConfig, *args, **kwargs) -> None:
        super().__init__(config)
        provider = config.provider
        if not provider:
            raise ValueError(
                "Must set provider in order to initialize `OllamaCompletionProvider`."
            )

        # Set up the Ollama client
        self.base_url = os.getenv(
            "OLLAMA_API_BASE", "http://localhost:11434"
        ).replace("/v1", "")
        logger.info(f"Using Ollama API base URL: {self.base_url}")
        self.client: Client = Client(host=self.base_url)
        self.aclient: AsyncClient = AsyncClient(host=self.base_url)
        self.converter = MessageConverter()

    @staticmethod
    def _get_base_args(generation_config: GenerationConfig) -> dict:
        """
        Convert R2R GenerationConfig to Ollama API parameters
        """
        args = {
            "model": generation_config.model.split("/")[-1],
            "stream": generation_config.stream,
            "think": generation_config.extended_thinking,
            "options": {},
        }
        # Override "think" if the model doesn't match any of the patterns
        if not any(
            [
                any(
                    [
                        generation_config.model.startswith(pattern)
                        for pattern in OllamaCompletionProvider.thinking_models_patterns
                    ]
                )
            ]
        ):
            args["think"] = False

        # Add generation parameters
        args["options"]["temperature"] = generation_config.temperature
        args["options"]["top_p"] = generation_config.top_p
        args["options"]["num_predict"] = generation_config.max_tokens_to_sample

        logger.debug(
            f"Converting R2R GenerationConfig to Ollama API parameters: {generation_config} -> {args}"
        )

        return args

    def _format_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[Message]:
        """
        Format messages for Ollama API - keep native Ollama format
        """
        return self.converter.r2r_to_ollama(messages)

    def _convert_tools_to_ollama_format(
        self,
        tools: list[dict[str, Any]],
    ) -> list[dict]:
        """
        Convert OpenAI-style tools to Ollama's native format
        """
        return self.converter.ollama_tools_from_r2r(tools)

    def _format_response(self, response: ChatResponse) -> dict[str, Any]:
        """
        Format Ollama response to match OpenAI format for compatibility
        """
        return self.converter.ollama_response_to_r2r(response)

    def _format_chunk_response(self, chunk: ChatResponse) -> dict[str, Any]:
        """
        Format Ollama chunk response to match OpenAI format for compatibility
        """
        return self.converter.ollama_chunk_to_r2r(chunk)

    def _create_args_for_task(
        self,
        generation_config: GenerationConfig,
        messages: list[dict[str, Any]],
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        logger.debug(f"Creating args for messages: {messages}")
        formatted_messages = self._format_messages(messages)
        args = self._get_base_args(generation_config)

        logger.debug(f"Generation config tools: {generation_config.tools}")

        if generation_config.tools:
            converted_tools = self._convert_tools_to_ollama_format(
                generation_config.tools
            )
            args["tools"] = converted_tools
            logger.debug(f"Converted tools for Ollama: {converted_tools}")
        else:
            logger.warning("No tools provided in generation_config!")

        args["messages"] = formatted_messages
        args.update(kwargs)

        logger.debug(f"Final Ollama generation config: {args}")
        return args

    @staticmethod
    def _deduplicate_task_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Deduplicate assistant messages in a task to avoid duplicates in the Ollama API.
        When the message list has duplicates, it leaves the last duplicated message in the list.
        """
        return list(
            {
                m["content"]
                + str(random.random() if m["role"] != "assistant" else ""): m
                for m in messages
            }.values()
        )

    async def _execute_task(self, task: dict[str, Any]):
        """
        Execute an async task with the Ollama API
        """
        messages = self._deduplicate_task_messages(task["messages"])
        logger.debug(f"Executing task: {task}")
        generation_config = task["generation_config"]

        kwargs = task.get("kwargs", {})

        args = self._create_args_for_task(generation_config, messages, kwargs)

        logger.info(f"Executing async Ollama task with args: {args}")

        try:
            if args.get("stream", False):
                # For streaming, we need to return an async iterator
                return self._acompletion_stream(args)
            else:
                # For non-streaming, just return the response
                response: ChatResponse = await self.aclient.chat(**args)
                logger.info(f"Ollama response: {response}")
                return LLMChatCompletion(**self._format_response(response))
        except Exception as e:
            logger.error(f"Async Ollama task execution failed: {e!r}")
            raise

    def _execute_task_sync(self, task: dict[str, Any]):
        """
        Execute a sync task with the Ollama API
        """
        messages = task["messages"]
        generation_config = task["generation_config"]
        kwargs = task.get("kwargs", {})

        args = self._create_args_for_task(generation_config, messages, kwargs)

        logger.debug(f"Executing sync Ollama task with args: {args}")

        try:
            if args.get("stream", False):
                return self._completion_stream(args)
            else:
                # For non-streaming, just return the response
                response = self.client.chat(**args)
                return LLMChatCompletion(**self._format_response(response))
        except Exception as e:
            logger.error(f"Sync Ollama task execution failed: {e!r}")
            raise

    async def _acompletion_stream(
        self,
        args: dict[str, Any],
        **kwargs,
    ) -> AsyncGenerator[LLMChatCompletionChunk, None]:
        """
        Override the base method to handle streaming properly for Ollama
        """
        chunk_count = 0
        tool_call_chunks = 0

        try:
            chat_response: AsyncIterator[
                ChatResponse
            ] = await self.aclient.chat(**args)
            async for chunk in chat_response:
                chunk_count += 1

                # Check if this chunk has tool calls
                if (
                    hasattr(chunk.message, "tool_calls")
                    and chunk.message.tool_calls
                ):
                    tool_call_chunks += 1
                    logger.info(
                        f"FOUND TOOL CALL in chunk #{chunk_count}: {chunk.message.tool_calls}"
                    )

                try:
                    formatted_chunk = self._format_chunk_response(chunk)
                    if (
                        formatted_chunk
                    ):  # Only yield if we have a valid formatted chunk
                        chunk_obj = LLMChatCompletionChunk(**formatted_chunk)
                        logger.debug(
                            f"Yielding chunk #{chunk_count}: {chunk_obj}"
                        )
                        yield chunk_obj
                    else:
                        logger.warning(
                            f"Chunk #{chunk_count} produced empty formatted_chunk"
                        )
                except Exception as chunk_error:
                    logger.error(
                        f"Error formatting chunk #{chunk_count} {chunk}: {str(chunk_error)}"
                    )
                    # Continue processing other chunks instead of failing completely
                    continue

            logger.info(
                f"Stream completed. Total chunks: {chunk_count}, Tool call chunks: {tool_call_chunks}"
            )

        except Exception as e:
            logger.error(f"Async Ollama streaming execution failed: {e!r}")
            raise

    def _completion_stream(
        self,
        args: dict[str, Any],
        **kwargs,
    ) -> Generator[LLMChatCompletionChunk, None, None]:
        """
        Override the base method to handle streaming properly for Ollama
        """

        try:
            for chunk in self.client.chat(**args):
                logger.debug(f"Processing chunk: {chunk}")
                try:
                    formatted_chunk = self._format_chunk_response(chunk)
                    if (
                        formatted_chunk
                    ):  # Only yield if we have a valid formatted chunk
                        yield LLMChatCompletionChunk(**formatted_chunk)
                except Exception as chunk_error:
                    logger.error(
                        f"Error formatting chunk {chunk}: {str(chunk_error)}"
                    )
                    # Continue processing other chunks instead of failing completely
                    continue
        except Exception as e:
            logger.error(
                f"Sync Ollama streaming task execution failed: {e!r}"
            )
            raise
