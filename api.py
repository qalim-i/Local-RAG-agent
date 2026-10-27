import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from chat import build_chain, stream_answer
from retrieval import get_retriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["retriever"] = get_retriever()
    state["chain"] = build_chain()
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/health")
def health():
    vector_store = state["retriever"].vectorstore
    return {
        "status": "ok",
        "documents": vector_store._collection.count(),
        "embedding_model": config.EMBEDDING_MODEL,
        "chat_model": config.CHAT_MODEL,
    }


@app.post("/chat")
def chat(request: ChatRequest):
    def event_stream():
        try:
            for event in stream_answer(request.question, state["retriever"], state["chain"]):
                yield _sse(event)
        except Exception as e:
            yield _sse({"type": "error", "message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=str(config.BASE_DIR / "static"), html=True), name="static")
