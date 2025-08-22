from typing import Annotated

from pydantic import BaseModel
from r2r import Tool
from shared import AggregateSearchResult


class NoOpResult(BaseModel):
    message: str


class NoOp(Tool):
    """
    LLM tool that does nothing except return a message to the user.

    This tool is designed to break out of tool calling loops by allowing the LLM
    to directly return a message to the user when it is stuck in a tool calling spree.

    Key capabilities:
    - Provides a way for the LLM to directly respond to the user
    - Helps prevent unnecessary repeated tool calls
    - Simple pass-through mechanism for messages

    Usage considerations:
    - Use when you notice you're calling the same tools repeatedly
    - Use when you've already gathered all necessary information
    - Use when you need to provide a complex response that doesn't require additional tool calls
    """

    def __init__(self):
        super().__init__(
            name="no_op",
            description=(
                "Use this tool to provide a direct response to the user when no further or no at all"
                " tool calls are needed. "
                "Call this when you've already gathered all necessary information or notice you're "
                "repeatedly calling the same tools (you are stuck). "
                "It just provides a way to return a natural language message to the user. "
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Your response to show to the user. E.g. 'Thank you!' or 'Sorry, I can't help you.'"
                        ),
                    },
                },
                "required": ["reason"],
            },
            results_function=self.execute,
            llm_format_function=None,
        )

    async def execute(
        self,
        reason: Annotated[str, "Your response to show to the user"],
        *args,
        **kwargs,
    ) -> AggregateSearchResult:
        """
        This function does nothing.

        Call this function if you notice that you have already called the same
        tool multiple times or you have already called all the tools you need.
        The message you return will be shown to the user as your response to their
        request.

        Parameters
        ----------
        reason : str
            Your response to show to the user
        *args
            Additional positional arguments to pass
        **kwargs
            Additional keyword arguments to pass

        Returns
        -------
        AggregateSearchResult
            Contains the message to be shown to the user
        """
        result = AggregateSearchResult(generic_tool_result=[NoOpResult(message=reason)])

        context = self.context
        # Add to results collector if context is provided
        if context and hasattr(context, "search_results_collector"):
            context.search_results_collector.add_aggregate_result(result)

        return result
