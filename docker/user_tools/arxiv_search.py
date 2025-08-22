import asyncio
import os
import threading
from asyncio import Task
from typing import ClassVar
from uuid import UUID

import arxiv
from pydantic import BaseModel

from core import logger
from r2r import Tool, R2RAsyncClient
from shared import AggregateSearchResult
from shared.api.models import IngestionResponse, WrappedIngestionResponse
from .utils import async_timeout

ARXIV_SLEEP = 0.5
R2R_SLEEP = 0.5
ARXIV_TIMEOUT = 45  # seconds


class ArxivSearchResult(BaseModel):
    title: str
    id: str
    authors: str
    abstract: str | None = None
    error: str | None = None
    ingestion_response: IngestionResponse | None = None


class ArxivSearch(Tool):
    """
    LLM tool that enables searching and ingesting research papers from arXiv into the LLM's context or
    R2R database system.

    Key capabilities:
    - Performs parametrized searches on arXiv using queries
    - Supports different retrieval modes (title-only, summary, full PDF content)
    - Handles concurrent PDF downloads and ingestion into R2R
    - Integrates with R2R's document storage and processing pipeline
    - Returns structured paper metadata (titles, authors, abstracts)

    Implementation details:
    - Uses arxiv-python client for fetching papers (thread-safe with locks)
    - Implements rate-limiting via semaphores for R2R ingestion
    - Handles errors gracefully during ingestion
    - Integrates with search results collector if context provided

    Performance characteristics:
    - Concurrent ingestion limited by R2R_INGESTION_CONCURRENCY env var (default 10)
    - Download and processing of PDFs can take significant time
    - Returns results asynchronously to avoid blocking

    Usage considerations:
    - Caller should expect delays when retrieving full PDFs
    - New database entries require rerunning searches to find
    - Recommend informing users about processing duration
    """

    arxiv_client: ClassVar[arxiv.Client] = arxiv.Client()
    arxiv_lock: ClassVar[threading.Lock] = threading.Lock()

    r2r_client: ClassVar[R2RAsyncClient] = R2RAsyncClient()
    r2r_semaphore: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(
        int(os.getenv("R2R_INGESTION_CONCURRENCY", "10"))
    )

    def __init__(self):
        super().__init__(
            name="arxiv_search",
            description=(
                "Provides an opportunity to find research papers in a popular platform Arxiv. "
                "Allows fetching of partial information and also downloading and ingestion of documents to the DB. "
                "Ingestion takes time so documents are not immediately available for use. "
                "Use this tool to find papers and ingest them into the R2R system."
                "When downloading papers, use \"id_list\" parameter with papers' IDs and provide \"search_query\": null."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "search_query": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "null"}
                        ],
                        "description": """
                            Query string to perform the search. Supports:
                            - Boolean operators: AND, OR, NOT
                            - Field prefixes: ti: (title), au: (author), abs: (abstract)
                            - Grouping with parentheses ()
                            - Quotes for exact phrases "quantum computing"
                            - Math/equations encoded in TeX format
                            E.g. "quantum computing" AND (python OR java).
                            This should be unencoded. Use `au:del_maestro AND ti:checkerboard`,
                            not `au:del_maestro+AND+ti:checkerboard`.
                            When it is required to search for papers by ID, e.g. 1234.567890v1,
                            or to use "retrieve_mode": "full", use "id_list" parameter with papers' IDs
                            and provide "search_query": null.
                            """,
                    },
                    "id_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of paper IDs to search. "
                            "If provided, result will be narrowed and only these papers are returned. "
                            "IMPORTANT: When having multiple IDs that user is interested in, always provide them "
                            "all in a single list. Do NOT make separate requests for each ID to prevent multiple "
                            "calls. Example: ['2307.09288v1', '2106.04554']. "
                            "IMPORTANT: When using this parameter for \"retrieve_mode\": \"full\", set \"search_query\": null."
                            "Default is an empty array."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            "How many Arxiv papers to fetch. Minimum value is 1. Default value is 5. "
                            "Maximum value is 50. Gets overridden if \"id_list\" is provided."
                        ),
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": list(arxiv.SortCriterion.__members__.keys()),
                        "description": "How to sort the results. Default is Relevance.",
                    },
                    "retrieve_mode": {
                        "type": "string",
                        "enum": ["title", "summary", "full"],
                        "description": (
                            "Determines the mode of retrieval for the results. "
                            "Options are: 'title' (Returns only the titles of results), "
                            "'summary' (Returns titles and summaries of results), "
                            "'full' (Returns titles, summaries, and downloads full PDF. "
                            "Use always with papers ids in \"id_list\" and \"search_query\": null!)."
                            "Default is 'title'."
                            "IMPORTANT: For \"retrieve_mode\": \"full\", NEVER use \"search_query\" with a value other than null."
                            "Use \"id_list\" parameter instead of \"search_query\"."
                        ),
                    },
                },
                "required": [],
            },
            results_function=self.execute,
            llm_format_function=None,
        )

    @async_timeout(ARXIV_TIMEOUT)
    async def execute(
            self,
            search_query: str | None = None,
            id_list: list[str] | None = None,
            max_results: int = 5,
            sort_by: str = "Relevance",
            retrieve_mode: str = "title",
            *args,
            **kwargs,
    ) -> AggregateSearchResult:
        """Execute a search query using the arXiv search API and process the results.

        Parameters
        ----------
        search_query : str | None
            Query string to perform the search. Supports:
            - Boolean operators: AND, OR, NOT
            - Field prefixes: ti: (title), au: (author), abs: (abstract)
            - Grouping with parentheses ()
            - Quotes for exact phrases "quantum computing"
            - Math/equations encoded in TeX format
            E.g. "quantum computing" AND (python OR java).
            This should be unencoded. Use `au:del_maestro AND ti:checkerboard`,
            not `au:del_maestro+AND+ti:checkerboard`.
            When it is required to search for papers by ID, e.g. 1234.567890v1,
            or to use "retrieve_mode": "full", use "id_list" parameter with papers' IDs
            and provide "search_query": null.
        id_list : list[str] | None
            List of paper IDs to search.
            If provided, result will be narrowed and only these papers are returned.
            IMPORTANT: When having multiple IDs that user is interested in, always provide them
            all in a single list. Do NOT make separate requests for each ID to prevent multiple
            calls. Example: ['2307.09288v1', '2106.04554'].
            IMPORTANT: When using this parameter for "retrieve_mode": "full", set "search_query": null.
            Default is an empty array.
        max_results : int
            How many Arxiv papers to fetch. Minimum value is 1. Default value is 5.
            Maximum value is 50. Gets overridden if "id_list" is provided.
        sort_by : str
            How to sort the results. Default is Relevance.
        retrieve_mode : str
            Determines the mode of retrieval for the results.
            Options are: 'title' (Returns only the titles of results),
            'summary' (Returns titles and summaries of results),
            'full' (Returns titles, summaries, and downloads full PDF.
            Use always with papers ids in "id_list" and "search_query": null!).
            Default is 'title'.
            IMPORTANT: For "retrieve_mode": "full", NEVER use "search_query" with a value other than null.
            Use "id_list" parameter instead of "search_query".
        *args
            Additional positional arguments to pass
        **kwargs
            Additional keyword arguments to pass

        Returns
        -------
        AggregateSearchResult
            Contains the aggregated search results
        """
        sort_criterion = arxiv.SortCriterion[sort_by]
        max_results = max_results if search_query else len(id_list)
        search_query = search_query if search_query not in ("null", None) else ""
        query = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=sort_criterion,
            id_list=id_list or [],
        )
        logger.info(f"Executing arXiv search query: {query!r}.")
        try:
            logger.debug(
                f"Entering arXiv search lock. Lock state locked?: {self.arxiv_lock.locked()}"
            )
            with self.arxiv_lock:
                # Convert generator to list to avoid potential hanging
                arxiv_results: list[arxiv.Result] = list(
                    await asyncio.wait_for(
                        asyncio.to_thread(self.arxiv_client.results, query),
                        timeout=0.5 * max_results,
                    )
                )
            if not arxiv_results:
                logger.info("No papers found for arXiv search query.")
                return AggregateSearchResult(
                    generic_tool_result=[
                        ArxivSearchResult(
                            title="",
                            id="",
                            authors="",
                            abstract="",
                            error="No papers found for arXiv search query. Ensure query if formatted correctly.",
                            ingestion_response=None,
                        )
                    ]
                )
        except Exception as e:
            logger.error(f"Error executing arXiv search query: {e!r}")
            return AggregateSearchResult(
                generic_tool_result=[
                    ArxivSearchResult(
                        title="",
                        id="",
                        authors="",
                        abstract="",
                        error=repr(e),
                        ingestion_response=None,
                    )
                ]
            )

        titles: list[str] = []
        ids: list[str] = []
        abstracts: dict[str, str] = {}  # not always fetched
        authors: list[str] = []
        download_tasks: dict[str, Task] = {}  # not always fetched
        for result in arxiv_results:
            titles.append(result.title)
            paper_id = result.entry_id.split("/")[-1]
            logger.info(f"Processing arXiv paper {paper_id}: {result.title}.")
            ids.append(paper_id)
            authors.append(", ".join([author.name for author in result.authors]))
            if retrieve_mode == "summary":
                abstracts[paper_id] = result.summary
            if retrieve_mode == "full":
                abstracts[paper_id] = result.summary
                # Wrap download in a task with potential timeout handling
                logger.info(
                    f"Scheduling download of PDF for arXiv paper {paper_id}: {result.title}."
                )
                download_task = asyncio.create_task(
                    asyncio.wait_for(
                        asyncio.to_thread(
                            result.download_pdf,
                            filename=f"{paper_id}.pdf",
                            dirpath="/tmp/",
                        ),
                        timeout=ARXIV_TIMEOUT * 0.5,  # Per-paper download timeout
                    )
                )
                download_task.add_done_callback(
                    lambda t: logger.info(
                        f"Download of PDF for arXiv paper {paper_id}: {result.title} completed"
                        + (f" with error: {t.exception()}" if t.exception() else "")
                    )
                )
                download_tasks[paper_id] = download_task
                await asyncio.sleep(ARXIV_SLEEP)

        logger.info(
            f"Fetched {len(titles)} of papers for arXiv search query. Titles: {titles}."
        )

        content_paths: list[str] = await asyncio.gather(
            *download_tasks.values(), return_exceptions=True
        )

        # submit pdfs to r2r ingestion API
        ingestion_tasks = []
        for idx, path in enumerate(content_paths):
            if isinstance(path, Exception):
                logger.error(
                    f"Error downloading PDF for arXiv paper {ids[idx]}: {titles[idx]}: {path!r}"
                )
                continue
            async with self.r2r_semaphore:
                doc_task = asyncio.create_task(
                    self.r2r_client.documents.create(
                        file_path=path,
                    )
                )
                ingestion_tasks.append(doc_task)
                await asyncio.sleep(R2R_SLEEP)

        if not ingestion_tasks:
            result = AggregateSearchResult(
                generic_tool_result=[
                    ArxivSearchResult(
                        title=title,
                        id=ids[idx],
                        authors=authors[idx],
                        abstract=abstracts.get(ids[idx]),
                        ingestion_response=IngestionResponse(
                            message=f"Noting to ingest - retrieval mode is {retrieve_mode}.",
                            document_id=UUID(int=0),
                            task_id=UUID(int=0),
                        ),
                    )
                    for idx, title in enumerate(titles)
                ]
            )
        else:
            ingestion_responses: list[WrappedIngestionResponse] = await asyncio.gather(
                *ingestion_tasks, return_exceptions=True
            )

            results: list[ArxivSearchResult] = []
            for idx, (resp, pth) in enumerate(zip(ingestion_responses, content_paths)):
                if isinstance(resp, Exception):
                    logger.error(
                        f"Error submitting document to r2r ingestion API: {resp!r}"
                    )
                    results.append(
                        ArxivSearchResult(
                            title=titles[idx],
                            id=ids[idx],
                            authors=authors[idx],
                            abstract=None,
                            ingestion_response=IngestionResponse(
                                message=f"Noting to ingest - encountered an error {resp!r}.",
                                document_id=UUID(int=0),
                                task_id=UUID(int=0),
                            ),
                        )
                    )
                else:
                    results.append(
                        ArxivSearchResult(
                            title=titles[idx],
                            id=ids[idx],
                            authors=authors[idx],
                            abstract=abstracts.get(ids[idx]),
                            ingestion_response=resp.results,
                        )
                    )
                os.remove(pth)

            result = AggregateSearchResult(generic_tool_result=results)

        context = self.context
        # Add to results collector if context is provided
        if context and hasattr(context, "search_results_collector"):
            context.search_results_collector.add_aggregate_result(result)

        return result
