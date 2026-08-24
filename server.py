from fastmcp import FastMCP


mcp = FastMCP("calculator")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return a / b


@mcp.tool()
def modulo(a: float, b: float) -> float:
    """Return the remainder of a divided by b."""
    if b == 0:
        raise ValueError("Modulo by zero is not allowed")
    return a % b


if __name__ == "__main__":
    # Avoid print statements because MCP uses stdout.
    mcp.run(
        transport="stdio",
        show_banner=False,
    )