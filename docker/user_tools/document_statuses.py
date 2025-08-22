import asyncio
import os
from typing import ClassVar, Optional

from .utils import async_timeout
from r2r import Tool, R2RAsyncClient
from shared import AggregateSearchResult
from core import logger
from pydantic import BaseModel, Field
from datetime import datetime


DOCUMENT_STATUS_TIMEOUT = 5  # seconds
R2R_SLEEP = 0.5


class DocumentStatusResult(BaseModel):
    """Result model for document status checks."""

    id: str
    title: str = "Unknown"
    document_type: str = ""
    ingestion_status: str = ""
    extraction_status: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    size_in_bytes: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: dict = Field(default_factory=dict)
    error: Optional[str] = None


class DocumentStatuses(Tool):
    """
    LLM tool that enables checking the status of documents being processed or ingested
    into the R2R system.

    Key capabilities:
    - Check status of specific documents by ID
    - Get detailed information about document processing stages

    Implementation details:
    - Uses R2R API client for fetching document statuses
    - Thread-safe with semaphores for API rate limiting
    - Handles errors gracefully
    - Returns structured document status information

    Performance characteristics:
    - Rate-limited to avoid overwhelming the API
    - Returns results asynchronously

    Usage considerations:
    - Useful for checking if documents are ready for querying
    - Can be used to diagnose processing failures
    - Provides visibility into the document ingestion pipeline
    """

    r2r_client: ClassVar[R2RAsyncClient] = R2RAsyncClient()
    r2r_semaphore: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(
        int(os.getenv("R2R_STATUS_CONCURRENCY", "10"))
    )

    def __init__(self):
        super().__init__(
            name="document_statuses",
            description=(
                "Check the status of documents in the R2R system. Retrieve information about document "
                "processing stages including ingestion status, extraction status, document metadata, "
                "and other properties. This tool is useful for checking if documents are ready for querying, "
                "diagnosing processing failures, or getting general information about documents in the system."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document_ids": {
                        "oneOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"}
                        ],
                        "description": (
                            "Document IDs to check. Use null for all documents, or array for specific documents. "
                            "Examples: \"document_ids\": null (all documents), "
                            "\"document_ids\": [\"af756572-c407-5256-bb41-d9cac3ef272c\"] (specific document)"
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of documents to return (1-100, default 10)",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Number of documents to skip for pagination (default 0)",
                    },
                    "owner_only": {
                        "type": "boolean",
                        "description": "Return only user-owned documents (default false)",
                    },
                },
                "required": ["document_ids"],
            },
            results_function=self.execute,
            llm_format_function=None,
        )

    @async_timeout(DOCUMENT_STATUS_TIMEOUT)
    async def execute(
        self,
        document_ids: list[str] | None = None,
        limit: int = 10,
        offset: int = 0,
        owner_only: bool = False,
        *args,
        **kwargs,
    ) -> AggregateSearchResult:
        """Execute a query for document statuses.

        Parameters
        ----------
        document_ids : list[str] | None
            Document IDs to check. Use None/null for all documents, or provide a list for specific documents.
            Example: ['550e8400-e29b-41d4-a716-446655440000'] for specific documents, None for all documents.
        limit : int
            Maximum number of document statuses to return. Minimum value is 1. Default value is 10. 
            Maximum value is 100.
        offset : int
            Number of documents to skip before starting to return results. Useful for pagination. 
            Default is 0.
        owner_only : bool
            If true, only returns documents owned by the user, not all accessible documents. 
            Default is false.
        *args
            Additional positional arguments to pass
        **kwargs
            Additional keyword arguments to pass

        Returns
        -------
        AggregateSearchResult
            Contains the aggregated document status results
        """
        # Handle null (meaning all documents) or validate list
        if document_ids is not None and not isinstance(document_ids, list):
            raise ValueError(f"document_ids must be a list or null, got {type(document_ids)}")

        # Validate limits
        if limit < 1 or limit > 100:
            raise ValueError(f"limit must be between 1 and 100, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")

        logger.info(
            f"Executing document status check for {len(document_ids) if document_ids else 'all'} documents."
        )

        try:
            async with self.r2r_semaphore:
                # Convert string IDs to list if provided, None means all documents
                ids_param = None
                if document_ids is not None:
                    # Ensure all IDs are strings
                    ids_param = [str(doc_id) for doc_id in document_ids]

                # Call the R2R API to get document statuses
                documents_response = await self.r2r_client.documents.list(
                    ids=ids_param,
                    limit=limit,
                    offset=offset,
                    include_summary_embeddings=False,
                    owner_only=owner_only,
                )

                document_results = documents_response.results
                logger.info(f"Retrieved {len(document_results)} document statuses.")

                # Create a DocumentStatusResult for each document
                results = []
                for doc in document_results:
                    status_result = DocumentStatusResult(
                        id=str(doc.id),
                        title=doc.title or "Unknown",
                        document_type=str(doc.document_type),
                        ingestion_status=str(doc.ingestion_status),
                        extraction_status=str(doc.extraction_status),
                        created_at=doc.created_at,
                        updated_at=doc.updated_at,
                        size_in_bytes=doc.size_in_bytes,
                        total_tokens=doc.total_tokens,
                        metadata=doc.metadata,
                    )
                    results.append(status_result)

                return AggregateSearchResult(generic_tool_result=results)

        except Exception as e:
            logger.error(f"Error retrieving document statuses: {e!r}")
            return AggregateSearchResult(
                generic_tool_result=[
                    DocumentStatusResult(
                        id="",
                        title="",
                        document_type="",
                        ingestion_status="",
                        extraction_status="",
                        error=repr(e),
                    )
                ]
            )
