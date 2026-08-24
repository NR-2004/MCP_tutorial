import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastmcp import Client
from gen_ai_hub.proxy.native.openai import OpenAI
from pydantic import BaseModel

load_dotenv()
from aicore_client import proxy_client

DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID")
SERVER_PATH = Path(__file__).with_name("server.py").resolve()
RESPONSE_DIR = Path(__file__).with_name("responses")
RESPONSE_FILE = RESPONSE_DIR / "responses.json"

SYSTEM_PROMPT = """
You are a calculator assistant with access to calculator tools.

Rules:
1. Always use the available tools for mathematical operations.
2. Never calculate mathematical results yourself.
3. Identify the required operations from the user's natural-language prompt.
4. For dependent operations, call tools sequentially.
5. Wait for the previous tool result before requesting the next tool.
6. Use the previous tool output as the next tool input when required.
7. Do not guess intermediate results.
8. After completing all tool calls, return the final result clearly.

Example:
User: Add 2 and 5, then multiply the result by 10.
1. Call add with a=2, b=5. 2. Wait for result. 3. Call multiply(result, 10). 4. Return final result.
"""

state: dict = {}


def convert_mcp_tools(mcp_tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": getattr(t, "inputSchema", None)
                or getattr(t, "input_schema", None)
                or {"type": "object", "properties": {}},
            },
        }
        for t in mcp_tools
    ]


def save_raw_response(response) -> Path:
    """Append the exact response object returned by AI Core into one shared JSON file."""
    RESPONSE_DIR.mkdir(parents=True, exist_ok=True)

    for attr in ("model_dump", "dict"):
        if hasattr(response, attr):
            response_dict = getattr(response, attr)()
            break
    else:
        raise TypeError(f"Cannot serialize response type: {type(response)}")

    if RESPONSE_FILE.exists():
        history = json.loads(RESPONSE_FILE.read_text(encoding="utf-8"))
    else:
        history = []

    history.append(response_dict)

    RESPONSE_FILE.write_text(
        json.dumps(history, indent=4, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Response saved: {RESPONSE_FILE}")
    return RESPONSE_FILE


def extract_tool_result(result) -> str:
    data = getattr(result, "data", None)
    if data is not None:
        return data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)

    texts = [c.text for c in getattr(result, "content", []) if getattr(c, "text", None)]
    return "\n".join(texts) if texts else str(result)


async def execute_tool(mcp_client: Client, tool_call) -> str:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
        print(f"\nCalling tool : {name}\nArguments    : {args}")
        result = extract_tool_result(await mcp_client.call_tool(name, args))
        print(f"Tool result  : {result}")
        return result
    except Exception as error:
        err = {"tool": name, "error": str(error)}
        print(f"Tool error   : {err}")
        return json.dumps(err)


def create_assistant_message(message) -> dict:
    entry = {"role": "assistant", "content": message.content}
    if message.tool_calls:
        entry["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in message.tool_calls
        ]
    return entry


async def process_prompt(prompt: str, mcp_client: Client, llm: OpenAI, tools: list[dict]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for round_number in range(1, 16):
        print(f"\nLLM round    : {round_number}")
        response = llm.chat.completions.create(
            deployment_id=DEPLOYMENT_ID,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        save_raw_response(response)

        msg = response.choices[0].message
        messages.append(create_assistant_message(msg))

        if not msg.tool_calls:
            return msg.content or "The model did not return an answer."

        for tool_call in msg.tool_calls:
            output = await execute_tool(mcp_client, tool_call)
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": output})

    raise RuntimeError("Maximum number of LLM/tool-call rounds reached.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DEPLOYMENT_ID:
        raise ValueError("DEPLOYMENT_ID is missing from the .env file.")
    if not SERVER_PATH.exists():
        raise FileNotFoundError(f"MCP server was not found: {SERVER_PATH}")

    state["llm"] = OpenAI(proxy_client=proxy_client)

    async with Client(str(SERVER_PATH)) as mcp_client:
        state["mcp_client"] = mcp_client
        mcp_tools = await mcp_client.list_tools()
        state["tools"] = convert_mcp_tools(mcp_tools)
        print("Calculator MCP client started.")
        print("Available tools:", ", ".join(t.name for t in mcp_tools))
        print("FastAPI ready at http://0.0.0.0:8000/docs")
        yield

    state.clear()


app = FastAPI(title="Calculator MCP Agent", lifespan=lifespan)


class PromptRequest(BaseModel):
    prompt: str


class PromptResponse(BaseModel):
    answer: str


@app.post("/prompt", response_model=PromptResponse)
async def run_prompt(req: PromptRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        answer = await process_prompt(prompt, state["mcp_client"], state["llm"], state["tools"])
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    return PromptResponse(answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("client:app", host="0.0.0.0", port=8000, reload=False)