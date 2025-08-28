import logging
import os
from copy import copy
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAI

from core.base import (
    ChunkSearchResult,
    EmbeddingConfig,
    EmbeddingProvider,
)

logger = logging.getLogger()


class HugginfaceEmbeddingProvider(EmbeddingProvider):
    """
    Embeddings provider for Hugging Face text-embeddings-inference (TEI) using the OpenAI SDK.

    Configuration:
    - Environment variables:
      - HUGGINGFACE_EMBEDDINGS: base URL of the TEI server (e.g., http://localhost:8080)
      - HUGGINGFACE_API_BASE: fallback base URL (same semantics)
    - EmbeddingConfig:
      - provider must be "huggingface"
      - base_model: TEI model identifier to pass to the OpenAI client (e.g., "tei" or
        "Qwen/Qwen3-Embedding-0.6B"). If base_model contains slashes, only the last
        segment will be sent to the API to mirror patterns used elsewhere.
      - base_dimension is ignored by TEI.
      - rerank_model is optional and ignored by TEI (single-model deployment). If set, it will be accepted.
      - rerank_url optional; if not set, will use the same TEI base with /rerank.
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)

        if not config.provider:
            raise ValueError(
                "Must set provider in order to initialize `HugginfaceEmbeddingProvider`."
            )
        if config.provider != "huggingface":
            raise ValueError(
                "HugginfaceEmbeddingProvider must be initialized with provider `huggingface`."
            )

        # TEI base URL
        base = os.getenv("HUGGINGFACE_EMBEDDINGS") or os.getenv("HUGGINGFACE_API_BASE")
        if not base:
            raise ValueError(
                "HUGGINGFACE_EMBEDDINGS or HUGGINGFACE_API_BASE environment variable must be set to the TEI server base URL (e.g., http://localhost:8080)."
            )


        # TEI with OpenAI SDK expects base_url ending at /v1
        base_url = base.rstrip("/") + "/v1"
        self.client = OpenAI(base_url=base_url)
        self.async_client = AsyncOpenAI(base_url=base_url)

        # Assert availability
        # try:
        #     self.client.embeddings.create(input=["test"], model="tei")
        #     logger.info(
        #         f"Successfully connected to Huggingface TEI server at {base_url}."
        #     )
        # except Exception as e:
        #     raise ValueError(
        #         f"Error initializing HuggingfaceEmbeddingProvider: {e!r}"
        #     ) from e

        # Model handling: mirror HF chat provider behavior by trimming namespace
        self.base_model = (
            config.base_model.split("/")[-1] if config.base_model else "tei"
        )
        self.base_dimension = config.base_dimension  # Not used by TEI

        # Optional batching via config
        self.batch_size = config.batch_size or 32

        # Rerank endpoint
        self.rerank_url = (
            config.rerank_url
            or os.getenv("HUGGINGFACE_API_BASE")
            or base
        )
        if self.rerank_url:
            self.rerank_url = self.rerank_url.rstrip("/") + "/rerank"

        # Persistent HTTPX clients for rerank endpoints
        # These clients are stored at the instance level to avoid re-creating them per request
        self._httpx_client = httpx.Client()
        self._httpx_async_client = httpx.AsyncClient()

    def _get_embedding_kwargs(self, **kwargs):
        embedding_kwargs: dict[str, Any] = {
            "model": self.base_model or "tei",
        }
        embedding_kwargs.update(kwargs)
        return embedding_kwargs

    async def _execute_task(self, task: dict[str, Any]) -> list[list[float]]:
        texts = task["texts"]
        kwargs = self._get_embedding_kwargs(**task.get("kwargs", {}))
        try:
            # The TEI OpenAI-compatible API supports batching natively.
            response = await self.async_client.embeddings.create(
                input=texts,
                **kwargs,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            error_msg = f"Error getting embeddings from Huggingface TEI: {e!r}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def _execute_task_sync(self, task: dict[str, Any]) -> list[list[float]]:
        texts = task["texts"]
        kwargs = self._get_embedding_kwargs(**task.get("kwargs", {}))
        try:
            response = self.client.embeddings.create(
                input=texts,
                **kwargs,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            error_msg = f"Error getting embeddings from Huggingface TEI: {e!r}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    async def async_get_embedding(
        self,
        text: str,
        stage: EmbeddingProvider.Step = EmbeddingProvider.Step.BASE,
        **kwargs,
    ) -> list[float]:
        if stage != EmbeddingProvider.Step.BASE:
            raise ValueError(
                "HugginfaceEmbeddingProvider only supports base embedding stage."
            )
        task = {"texts": [text], "stage": stage, "kwargs": kwargs}
        result = await self._execute_with_backoff_async(task)
        return result[0]

    def get_embedding(
        self,
        text: str,
        stage: EmbeddingProvider.Step = EmbeddingProvider.Step.BASE,
        **kwargs,
    ) -> list[float]:
        if stage != EmbeddingProvider.Step.BASE:
            raise ValueError(
                "HugginfaceEmbeddingProvider only supports base embedding stage."
            )
        task = {"texts": [text], "stage": stage, "kwargs": kwargs}
        result = self._execute_with_backoff_sync(task)
        return result[0]

    async def async_get_embeddings(
        self,
        texts: list[str],
        stage: EmbeddingProvider.Step = EmbeddingProvider.Step.BASE,
        **kwargs,
    ) -> list[list[float]]:
        if stage != EmbeddingProvider.Step.BASE:
            raise ValueError(
                "HugginfaceEmbeddingProvider only supports base embedding stage."
            )
        task = {"texts": texts, "stage": stage, "kwargs": kwargs}
        return await self._execute_with_backoff_async(task)

    def get_embeddings(
        self,
        texts: list[str],
        stage: EmbeddingProvider.Step = EmbeddingProvider.Step.BASE,
        **kwargs,
    ) -> list[list[float]]:
        if stage != EmbeddingProvider.Step.BASE:
            raise ValueError(
                "HugginfaceEmbeddingProvider only supports base embedding stage."
            )
        task = {"texts": texts, "stage": stage, "kwargs": kwargs}
        return self._execute_with_backoff_sync(task)

    def rerank(
        self,
        query: str,
        results: list[ChunkSearchResult],
        stage: EmbeddingProvider.Step = EmbeddingProvider.Step.RERANK,
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        if not self.rerank_url:
            return results[:limit]

        texts = [r.text for r in results]
        payload = {
            "query": query,
            "texts": texts,
            # TEI defaults raw_scores=false; return_text allows server to include text
            "raw_scores": False,
            "return_text": False,
        }
        headers = {"Content-Type": "application/json"}
        try:
            resp = self._httpx_client.post(self.rerank_url, json=payload, headers=headers)
            return self._process_rerank_response(resp, limit, results)
        except Exception as e:
            logger.error(f"Error during TEI reranking: {e}")
            return results[:limit]

    async def arerank(
        self,
        query: str,
        results: list[ChunkSearchResult],
        stage: EmbeddingProvider.Step = EmbeddingProvider.Step.RERANK,
        limit: int = 10,
    ) -> list[ChunkSearchResult]:
        if not self.rerank_url:
            return results[:limit]

        texts = [r.text for r in results]
        payload = {
            "query": query,
            "texts": texts,
            "raw_scores": False,
            "return_text": False,
        }
        headers = {"Content-Type": "application/json"}
        try:
            resp = await self._httpx_async_client.post(self.rerank_url, json=payload, headers=headers)
            return self._process_rerank_response(resp, limit, results)
        except Exception as e:
            logger.error(f"Error during async TEI reranking: {e}")
            return results[:limit]


    @staticmethod
    def _process_rerank_response(
            resp: httpx.Response,
            limit: int,
            results: list[ChunkSearchResult],
    ) -> list[ChunkSearchResult]:
        resp.raise_for_status()
        reranked = resp.json()
        scored_results: list[ChunkSearchResult] = []
        for item in reranked:
            idx = item.get("index")
            if idx is None or idx >= len(results):
                continue
            cloned = copy(results[idx])
            cloned.score = item.get("score", cloned.score)
            scored_results.append(cloned)
        return scored_results[:limit]
