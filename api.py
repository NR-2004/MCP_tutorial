from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from client import invoke_agent

app = FastAPI(title="Math MCP Agent")


class PromptRequest(BaseModel):
    prompt: str


class PromptResponse(BaseModel):
    answer: str


@app.post("/calculate", response_model=PromptResponse)
async def calculate(request: PromptRequest) -> PromptResponse:
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    try:
        answer = await invoke_agent(request.prompt)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    return PromptResponse(answer=answer)