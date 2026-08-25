import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastmcp import Client
from gen_ai_hub.proxy.native.openai import OpenAI

from ai_core_client import proxy_client
from utils.response_store import save_raw_responses


load_dotenv()

DEPLOYMENT_ID = os.getenv("DEPLOYMENT_ID")
SERVER_PATH = Path(__file__).with_name("server.py").resolve()

SYSTEM_PROMPT = """
You are a maths assistant. You must access the provided tools and perform the
calculation.

RULES:
1. Use the provided tools in the server for every arithmetic operation.
2. Do not calculate by yourself.
3. Understand the required operations given in natural language.
4. Call the tools sequentially.
5. Call the next tool only after execution of the previous tool.
6. Do not guess an intermediate answer or the final answer.
7. Give the result in a clear form.
""".strip()


def require_deployment_id() -> str:
    if not DEPLOYMENT_ID:
        raise ValueError("DEPLOYMENT_ID is missing from the environment.")
    return DEPLOYMENT_ID


def mcp_tools_to_openai(tools: list[Any]) -> list[dict[str, Any]]:
    definitions = []
    for tool in tools:
        parameters = getattr(tool, "inputSchema", None)
        if parameters is None:
            parameters = getattr(tool, "input_schema", None)
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": parameters or {"type": "object", "properties": {}},
                },
            }
        )
    return definitions


def tool_result_text(result: Any) -> str:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)

    parts = []
    for item in getattr(result, "content", []):
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
    return "\n".join(parts) if parts else str(result)


async def invoke_agent(user_prompt: str) -> str:
    # The SAP SDK's OpenAI constructor accepts the proxy client. A specific
    # AI Core deployment is selected with `deployment_id` on the completion.
    llm = OpenAI(proxy_client=proxy_client)
    deployment_id = require_deployment_id()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    execution_responses: list[Any] = []  # all raw LLM responses for this one execution

    async with Client(str(SERVER_PATH)) as mcp_client:
        available_tools = await mcp_client.list_tools()
        openai_tools = mcp_tools_to_openai(available_tools)

        while True:
            response = llm.chat.completions.create(
                deployment_id=deployment_id,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                parallel_tool_calls=False,
            )
            execution_responses.append(response)

            assistant_message = response.choices[0].message
            messages.append(
                assistant_message.model_dump(exclude_none=True)
            )

            if not assistant_message.tool_calls:
                saved_path = save_raw_responses(execution_responses)
                print(f"Raw LLM responses ({len(execution_responses)}) saved to: {saved_path}")
                return assistant_message.content or ""

            for tool_call in assistant_message.tool_calls:
                arguments = json.loads(
                    tool_call.function.arguments or "{}"
                )

                result = await mcp_client.call_tool(
                    tool_call.function.name,
                    arguments,
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result_text(result),
                    }
                )


async def main() -> None:
    print("Math MCP Agent (type 'exit' to stop)")
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() in {"exit", "quit"}:
            break
        if not prompt:
            continue
        try:
            answer = await invoke_agent(prompt)
            print("Assistant:", answer)
        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    asyncio.run(main())