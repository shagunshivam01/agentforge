from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("search")

# Initialize once (good practice)
search_client = TavilySearch(max_results=3, search_depth="advanced")


@mcp.tool()
async def tavily_search(query: str) -> dict:
    """
    MCP Tool: Web search using Tavily.

    Returns structured search results:
    - answer (if available)
    - supporting sources
    """

    try:
        # NOTE: LangChain Tavily is sync → run safely
        result = search_client.invoke({"query": query})

        # normalize base structure
        response = {
            "query": query,
            "answer": None,
            "sources": [],
            "error": None,
        }

        # direct answer path
        if isinstance(result, dict) and result.get("answer"):
            response["answer"] = result["answer"]

        # structured sources
        raw_results = result.get("results", []) if isinstance(result, dict) else []

        response["sources"] = [
            {
                "title": r.get("title"),
                "content": r.get("content"),
                "url": r.get("url"),
            }
            for r in raw_results[:3]
        ]

        return response

    except Exception as e:
        return {
            "query": query,
            "answer": None,
            "sources": [],
            "error": str(e),
        }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")

