import asyncio
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Optional

from core import logger
from pydantic import BaseModel
from uuid import UUID, uuid5, NAMESPACE_URL
from r2r import Tool, R2RAsyncClient, R2RException
from shared import AggregateSearchResult
from shared.utils.base_utils import generate_default_user_collection_id
from shared.api.models import IngestionResponse, WrappedIngestionResponse

from .utils import async_timeout


class GitPlatform(StrEnum):
    """Supported Git hosting platforms."""
    GITHUB = "github"
    BITBUCKET = "bitbucket"


# Defaults and timeouts
GIT_TIMEOUT = int(os.getenv("R2R_GIT_TIMEOUT", "120"))  # seconds
R2R_SLEEP = float(os.getenv("R2R_INGESTION_SLEEP", "0.25"))
MAX_CONCURRENCY = int(os.getenv("R2R_INGESTION_CONCURRENCY", "10"))
DEFAULT_DEST = os.getenv("R2R_GIT_DEST", "/tmp/r2r_repos")
AVAILABLE_REPOS = os.getenv("R2R_AVAILABLE_REPOS", "").split(",")


@dataclass
class _RepoInfo:
    repo_url: str
    branch: Optional[str]
    local_dir: Path
    commit_hash: Optional[str] = None


class GitIngestResult(BaseModel):
    file_path: str
    repo_url: str
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    ingestion_response: Optional[IngestionResponse] = None
    error: Optional[str] = None


class GitRepoIngest(Tool):
    """
    Clone (if missing) or pull (if exists) a Git repository and ingest all Markdown (.md) files
    into the R2R database.

    Capabilities:
    - Clone or update a repository from a generic Git host (GitHub/Bitbucket/etc.)
    - Select branch to checkout
    - Find and ingest all Markdown files using R2R's ingestion API

    Usage notes:
    - Ensure R2R API base URL and auth are configured as for other tools.
    - For large repositories, ingestion is concurrent and rate-limited by env R2R_INGESTION_CONCURRENCY.
    - You can exclude files via exclude_glob.
    """

    r2r_client: ClassVar[R2RAsyncClient] = R2RAsyncClient()
    r2r_semaphore: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(MAX_CONCURRENCY)

    def __init__(self):
        super().__init__(
            name="git_repo_ingest",
            description=(
                "Clone or update a Git repository and ingest all Markdown (.md) files into R2R. "
                "Use this to sync documentation / blogpost repos (GitHub/Bitbucket/etc.) with the R2R knowledge base. "
                "You can filter files using include_glob and exclude_glob patterns."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "repo_url": {
                        "type": "string",
                        "description": f"Git repository URL (HTTPS). Must be one of: {AVAILABLE_REPOS}.",
                    },
                    "branch": {
                        "oneOf": [{"type": "string"}, {"type": "null"}],
                        "description": "Branch to checkout (default repo default)",
                    },
                    "exclude_glob": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "description": "Glob(s) to exclude (optional)",
                    },
                    "include_glob": {
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "description": "Glob(s) to include (optional). If specified, only files matching these patterns will be considered.",
                    },
                    "metadata": {
                        "oneOf": [
                            {"type": "object"},
                            {"type": "null"}
                        ],
                        "description": "Optional metadata to attach to each document",
                    },
                },
                "required": ["repo_url"],
            },
            results_function=self.execute,
            llm_format_function=None,
        )

    @staticmethod
    def _repo_dir_from_url(repo_url: str) -> str:
        name = repo_url.rstrip("/").split("/")[-1]
        # strip .git if present
        if name.endswith(".git"):
            name = name[:-4]
        # sanitize to filesystem-safe
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
        return name

    @staticmethod
    def _is_git_repo(path: Path) -> bool:
        return (path / ".git").exists()

    @staticmethod
    def _detect_repo_host(repo_url: str) -> Optional[GitPlatform]:
        """Detect the Git hosting service from the repository URL."""
        repo_url_lower = repo_url.lower()
        if "github.com" in repo_url_lower:
            return GitPlatform.GITHUB
        elif "bitbucket.org" in repo_url_lower:
            return GitPlatform.BITBUCKET
        return None

    @staticmethod
    def _inject_github_auth(repo_url: str) -> str:
        """Inject GitHub authentication token into repository URL."""
        token = os.getenv("GITHUB_API_TOKEN")
        if not token:
            return repo_url

        # Convert https://github.com/user/repo.git to https://token@github.com/user/repo.git
        if repo_url.startswith("https://github.com/"):
            return repo_url.replace("https://github.com/", f"https://{token}@github.com/")
        elif repo_url.startswith("https://www.github.com/"):
            return repo_url.replace("https://www.github.com/", f"https://{token}@www.github.com/")
        return repo_url

    @staticmethod
    def _inject_bitbucket_auth(repo_url: str) -> str:
        """Inject Bitbucket authentication token into repository URL."""
        token = os.getenv("BITBUCKET_API_TOKEN")
        if not token:
            return repo_url

        # Convert https://bitbucket.org/workspace/repo.git to https://x-token-auth:token@bitbucket.org/workspace/repo.git
        if repo_url.startswith("https://bitbucket.org/"):
            return repo_url.replace("https://bitbucket.org/", f"https://x-token-auth:{token}@bitbucket.org/")
        elif repo_url.startswith("https://www.bitbucket.org/"):
            return repo_url.replace("https://www.bitbucket.org/", f"https://x-token-auth:{token}@www.bitbucket.org/")
        return repo_url

    @staticmethod
    def _inject_auth_token(repo_url: str) -> str:
        """Inject authentication token into repository URL if available."""
        host = GitRepoIngest._detect_repo_host(repo_url)

        if host == GitPlatform.GITHUB:
            return GitRepoIngest._inject_github_auth(repo_url)
        elif host == GitPlatform.BITBUCKET:
            return GitRepoIngest._inject_bitbucket_auth(repo_url)

        return repo_url

    @staticmethod
    def _run_git(cmd: list[str], cwd: Optional[Path] = None, timeout: int = GIT_TIMEOUT) -> str:
        git_cmd = ["git"] + cmd
        logger.info(f"Running git command: {' '.join(shlex.quote(c) for c in git_cmd)} in {cwd}")

        # Check if git command exists
        try:
            subprocess.run(["which", "git"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("Git command not found. Please ensure git is installed in the container.")

        # Check if working directory exists
        if cwd and not cwd.exists():
            raise RuntimeError(f"Working directory {cwd} does not exist")

        try:
            proc = subprocess.run(
                git_cmd,
                cwd=str(cwd) if cwd else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
        except FileNotFoundError as e:
            logger.error(f"Command not found: {' '.join(git_cmd)}. Error: {e}")
            raise RuntimeError(f"Git command not found: {e}")
        except subprocess.TimeoutExpired as e:
            logger.error(f"Git command timed out after {timeout}s: {' '.join(git_cmd)}")
            raise RuntimeError(f"Git command timed out: {e}")

        if proc.returncode != 0:
            logger.error(f"Git command failed: {' '.join(git_cmd)}\nSTDERR: {proc.stderr.strip()}")
            raise RuntimeError(f"Git command failed: {' '.join(git_cmd)}\nSTDERR: {proc.stderr.strip()}")
        return proc.stdout.strip()

    @classmethod
    async def _clone_or_update_repo(
            cls, repo_url: str, branch: Optional[str]
    ) -> _RepoInfo:
        base_dir = Path(DEFAULT_DEST)
        base_dir.mkdir(parents=True, exist_ok=True)
        repo_dir = base_dir / cls._repo_dir_from_url(repo_url)

        # Get authenticated URL for clone/fetch operations
        auth_repo_url = cls._inject_auth_token(repo_url)

        if repo_dir.exists() and cls._is_git_repo(repo_dir):
            # update
            logger.info(f"Updating existing repo at {repo_dir}")

            # Update remote URL with authentication if token is available and URL changed
            if auth_repo_url != repo_url:
                try:
                    await asyncio.to_thread(cls._run_git, ["remote", "set-url", "origin", auth_repo_url], repo_dir)
                except Exception as e:
                    logger.warning(f"Failed to update remote URL with auth token: {e!r}")

            await asyncio.to_thread(cls._run_git, ["fetch", "--all"], repo_dir)
            if branch:
                # ensure branch exists locally
                await asyncio.to_thread(cls._run_git, ["checkout", branch], repo_dir)
                await asyncio.to_thread(cls._run_git, ["pull", "origin", branch], repo_dir)
            else:
                await asyncio.to_thread(cls._run_git, ["pull"], repo_dir)
        else:
            # clone
            logger.info(f"Cloning repo {repo_url} into {repo_dir}")
            cmd = ["clone", auth_repo_url]
            if branch:
                cmd = ["clone", "--branch", branch, auth_repo_url]
            await asyncio.to_thread(cls._run_git, cmd, base_dir)

        # get commit hash
        try:
            commit = await asyncio.to_thread(
                cls._run_git, ["rev-parse", "HEAD"], repo_dir
            )
        except Exception as e:
            logger.warning(f"Unable to get commit hash for {repo_dir}: {e!r}")
            commit = None

        return _RepoInfo(repo_url=repo_url, branch=branch, local_dir=repo_dir, commit_hash=commit)

    @staticmethod
    def _match_excludes(path: Path, exclude_glob: Optional[list[str] | str]) -> bool:
        if not exclude_glob:
            return False
        patterns = exclude_glob if isinstance(exclude_glob, list) else [exclude_glob]
        for pat in patterns:
            if path.match(pat):
                return True
        return False

    @staticmethod
    def _match_includes(path: Path, include_glob: Optional[list[str] | str]) -> bool:
        if not include_glob:
            return True  # If no include patterns specified, include all files
        patterns = include_glob if isinstance(include_glob, list) else [include_glob]
        for pat in patterns:
            if path.match(pat):
                return True
        return False


    async def _assign_access_to_existing(self, document_id: UUID) -> None:
        """Ensure the current user has access to the existing document by adding the
        document to the user's default collection.
        """
        # Get current user id
        try:
            me = await self.r2r_client.users.me()
            user_uuid = me.results.id
        except Exception as e:
            logger.error(f"Failed to retrieve current user for permission assignment: {e!r}")
            return

        # Compute default collection ID deterministically for the user
        try:
            default_collection_id = str(generate_default_user_collection_id(user_uuid))
        except Exception as e:
            logger.error(f"Failed to compute default collection id for user {user_uuid}: {e!r}")
            return

        # Add the document to the user's default collection
        try:
            await self.r2r_client.collections.add_document(default_collection_id, str(document_id))
        except R2RException as e:
            msg_l = (e.message or "").lower()
            if e.status_code == 409 and ("already" in msg_l or "exists" in msg_l):
                logger.debug(
                    f"Document {document_id} already present in user's default collection {default_collection_id}"
                )
                return
            logger.info(
                f"Could not add existing document {document_id} to user's default collection {default_collection_id}: {e!r}"
            )
        except Exception as e:
            logger.info(
                f"Unexpected error while adding document {document_id} to default collection {default_collection_id}: {e!r}"
            )

    @async_timeout(int(os.getenv("R2R_GIT_INGEST_TIMEOUT", "600")))
    async def execute(
            self,
            repo_url: str,
            branch: Optional[str] = None,
            exclude_glob: Optional[list[str] | str] = None,
            include_glob: Optional[list[str] | str] = None,
            metadata: Optional[dict] = None,
            *args,
            **kwargs,
    ) -> AggregateSearchResult:
        # Validate if repo_url is in AVAILABLE_REPOS
        if AVAILABLE_REPOS and repo_url not in AVAILABLE_REPOS:
            error_msg = f"Repository {repo_url} is not in the list of available repositories: {AVAILABLE_REPOS}"
            logger.error(error_msg)
            return AggregateSearchResult(
                generic_tool_result=[
                    GitIngestResult(
                        file_path="",
                        repo_url=repo_url,
                        branch=branch,
                        commit_hash=None,
                        error=error_msg,
                    )
                ]
            )

        # Step 1: clone or update the repo
        try:
            repo_info = await self._clone_or_update_repo(repo_url, branch)
        except Exception as e:
            logger.error(f"Failed to clone or update repo {repo_url}: {e!r}")
            return AggregateSearchResult(
                generic_tool_result=[
                    GitIngestResult(
                        file_path="",
                        repo_url=repo_url,
                        branch=branch,
                        commit_hash=None,
                        error=repr(e),
                    )
                ]
            )

        # Step 2: find markdown files
        # Already ingested files are effectively skipped by using a deterministic document ID
        # per repo_url@commit:relative_path and catching R2RException 409 conflicts during ingestion.
        md_files = [
            p for p in repo_info.local_dir.glob("**/*.md")
            if p.is_file() and self._match_includes(p, include_glob) and not self._match_excludes(p, exclude_glob)
        ]
        md_files += [
            p for p in repo_info.local_dir.glob("**/*.markdown")
            if p.is_file() and self._match_includes(p, include_glob) and not self._match_excludes(p, exclude_glob)
        ]

        logger.info(
            f"Found {len(md_files)} markdown files in {repo_info.repo_url} given parameters: branch={branch}, "
            f"exclude_glob={exclude_glob}, include_glob={include_glob}."
        )
        if not md_files:
            logger.info(f"No markdown files found in {repo_info.local_dir}.")
            return AggregateSearchResult(
                generic_tool_result=[
                    GitIngestResult(
                        file_path="",
                        repo_url=repo_info.repo_url,
                        branch=repo_info.branch,
                        commit_hash=repo_info.commit_hash,
                        ingestion_response=IngestionResponse(
                            message="No markdown files found to ingest.",
                            document_id=UUID(int=0),
                            task_id=UUID(int=0),
                        ),
                    )
                ]
            )

        # Step 3: ingest files concurrently
        ingestion_tasks: list[asyncio.Task] = []
        file_paths: list[str] = []
        document_ids: list[UUID] = []
        for fpath in md_files:
            file_paths.append(str(fpath))
            # Deterministic ID to detect if this exact file revision was already ingested
            rel_path = str(fpath.relative_to(repo_info.local_dir))
            deterministic_str = f"{repo_info.repo_url}@{repo_info.commit_hash}:{rel_path}"
            doc_id = uuid5(NAMESPACE_URL, deterministic_str)
            document_ids.append(doc_id)
            async with self.r2r_semaphore:
                doc_metadata = {
                    "source": "git",
                    "repo_url": repo_info.repo_url,
                    "repo_branch": repo_info.branch,
                    "repo_commit": repo_info.commit_hash,
                    "repo_path": rel_path,
                }
                if metadata:
                    # user-provided metadata overrides defaults on key conflicts
                    doc_metadata.update(metadata)
                task = asyncio.create_task(
                    self.r2r_client.documents.create(
                        file_path=str(fpath),
                        id=str(doc_id),
                        metadata=doc_metadata,
                    )
                )
                ingestion_tasks.append(task)
                await asyncio.sleep(R2R_SLEEP)

        responses = await asyncio.gather(*ingestion_tasks, return_exceptions=True)

        results: list[GitIngestResult] = []
        for idx, resp in enumerate(responses):
            fpath = file_paths[idx]
            doc_id = document_ids[idx] if idx < len(document_ids) else UUID(int=0)
            if isinstance(resp, Exception):
                # If already exists, mark as skipped rather than error
                if isinstance(resp, R2RException):
                    msg_l = (resp.message or "").lower()
                    if resp.status_code == 409 and (
                            "already exists" in msg_l or "already ingested" in msg_l
                    ):
                        try:
                            await self._assign_access_to_existing(doc_id)
                        except Exception as e:
                            logger.warning(f"Failed to assign access for existing document {doc_id}: {e!r}")
                        results.append(
                            GitIngestResult(
                                file_path=fpath,
                                repo_url=repo_info.repo_url,
                                branch=repo_info.branch,
                                commit_hash=repo_info.commit_hash,
                                ingestion_response=IngestionResponse(
                                    message="Skipped - document already ingested. Added to user's default collection.",
                                    document_id=doc_id,
                                    task_id=UUID(int=0),
                                ),
                            )
                        )
                        continue
                logger.error(f"Error ingesting file {fpath}: {resp!r}")
                results.append(
                    GitIngestResult(
                        file_path=fpath,
                        repo_url=repo_info.repo_url,
                        branch=repo_info.branch,
                        commit_hash=repo_info.commit_hash,
                        error=repr(resp),
                    )
                )
            else:
                wrapped: WrappedIngestionResponse = resp  # type: ignore
                results.append(
                    GitIngestResult(
                        file_path=fpath,
                        repo_url=repo_info.repo_url,
                        branch=repo_info.branch,
                        commit_hash=repo_info.commit_hash,
                        ingestion_response=wrapped.results,
                    )
                )

        aggregate = AggregateSearchResult(generic_tool_result=results)

        # If context has a result collector, add it as other tools do
        context = self.context
        if context and hasattr(context, "search_results_collector"):
            context.search_results_collector.add_aggregate_result(aggregate)

        return aggregate
