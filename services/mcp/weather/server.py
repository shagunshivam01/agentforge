from mcp.server.fastmcp import FastMCP

mcp=FastMCP("weather")

@mcp.tool()
async def getWeather(location: str) -> str:
    """Get the weather by location"""
    return "It usually rains by the evening in Bangalore."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
