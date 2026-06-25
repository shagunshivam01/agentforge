from mcp.server.fastmcp import FastMCP
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

load_dotenv()

mcp = FastMCP("search")

# initialize once (IMPORTANT FIX)
search_tool = TavilySearch(max_results=3, search_depth="advanced")


@mcp.tool()
async def tavily_search(query: str) -> dict:
    """
    Perform web search using Tavily and return structured results.
    """

    try:
        result = search_tool.invoke({"query": query})

        # direct answer preferred
        if isinstance(result, dict) and result.get("answer"):
            return {
                "query": query,
                "answer": result["answer"],
                "sources": result.get("results", [])[:3],
            }

        # fallback: structured snippets
        sources = []
        for r in result.get("results", [])[:3]:
            sources.append({
                "title": r.get("title"),
                "content": r.get("content"),
                "url": r.get("url"),
            })

        return {
            "query": query,
            "answer": None,
            "sources": sources,
        }

    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "sources": []
        }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
