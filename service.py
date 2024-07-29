import bentoml
import httpx
import json
import time
import typing as t
import uuid
from fastapi import FastAPI, Response, Request
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse, JSONResponse
from typing import AsyncGenerator, Literal
from enum import Enum
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, Field


TOXIC_MODEL_ID = "martin-ha/toxic-comment-model"

class ModelName(str, Enum):
    gpt3 = "gpt-3.5-turbo"
    gpt4 = "gpt-4o"
    mistral = "mistral"
    llama3_1 = "llama3.1"


MODEL_INFO = {
    # remote_model_name, remote_base_url
    "gpt-3.5-turbo": ("gpt-3.5-turbo", "https://api.openai.com/v1"),
    "gpt-4o": ("gpt-4o", "https://api.openai.com/v1"),
    "mistral": ("mistral", ""),
    "llama3.1": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "https://bentovllm-llama-3-1-8-b-insruct-service-7o5r-d3767914.mt-guc1.bentoml.ai/v1"),
}


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "bentoml"
    root: t.Optional[str] = None
    parent: t.Optional[str] = None
    max_model_len: t.Optional[int] = None


class ModelList(BaseModel):
    object: str = "list"
    data: t.List[ModelCard] = Field(default_factory=list)


class GeneralChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: t.List[t.Dict[str, str]]
    model: ModelName
    stream: t.Optional[bool] = False


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: t.Optional[str] = None
    code: int


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def construct_cache_key(req: GeneralChatCompletionRequest) -> str:
    l = [req.model]
    for msg in req.messages:
        l.append(msg["role"])
        l.append(msg["content"])
    return "".join(l)


app = FastAPI()


@bentoml.mount_asgi_app(app, path="/v1")
@bentoml.service(
    traffic={
        "concurrency": 100,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-tesla-t4"
    },
)
class LLMGateway:

    def __init__(self):

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(TOXIC_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(TOXIC_MODEL_ID).to(device)
        self.pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)
        self.request_cache = {}

        # FastAPI endpoints definitions below:

        @app.get("/models")
        async def models():
            model_cards = []
            for model_name in MODEL_INFO:
                card = ModelCard(id=model_name)
                model_cards.append(card)

            models = ModelList(data=model_cards)
            return JSONResponse(content=models.model_dump())


        @app.post("/chat/completions")
        async def chat_request(request: GeneralChatCompletionRequest, raw_request:Request, response: Response):

            # return cached result if available
            if not request.stream:
                cache_key = construct_cache_key(request)
                if cache_key in self.request_cache:
                    response.body = self.request_cache[cache_key]
                    response.status_code = 200
                    return response

            # detect toxic messages
            safe = self.safe_detect(request.messages)
            if not safe:
                message = dict(
                    role="assistant",
                    content="toxic message detected!",
                    tool_calls=[],
                )

                choice = {"index": 0}
                if request.stream:
                    choice["delta"] = message
                else:
                    choice["message"] = message

                unsafe_response = dict(
                    id=f"chatcmpl-{random_uuid()}",
                    model="toxic_detector",
                    choices=[choice],
                )
                if request.stream:
                    return EventSourceResponse(
                        (i for i in [json.dumps(unsafe_response)]),
                        status_code=200,
                        media_type="text/event-stream",
                    )
                else:
                    return JSONResponse(
                        content=unsafe_response,
                        status_code=200,
                    )

            # route request to remote services
            remote_model_name, remote_base_url = MODEL_INFO[request.model]
            url = remote_base_url + "/chat/completions"
            client = httpx.AsyncClient()

            auth = raw_request._headers["authorization"]
            headers = {"authorization": auth}
            req_dict = request.dict()
            req_dict["model"] = remote_model_name

            remote_req = client.build_request(
                "POST", url, headers=headers, json=req_dict, timeout=300
            )
            remote_response = await client.send(remote_req, stream=True)
            content_type = remote_response.headers.get('content-type')

            if content_type.startswith("text/event-stream"):
                return EventSourceResponse(
                    remote_response.aiter_bytes(),
                    status_code=remote_response.status_code,
                    media_type=content_type,
                    background=BackgroundTask(remote_response.aclose)
                )

            else:
                await remote_response.aread()

                response.body = remote_response.content
                response.status_code = remote_response.status_code

                if not request.stream and remote_response.status_code == 200:
                    self.request_cache[cache_key] = remote_response.content

                return response


    def safe_detect(self, messages: t.List[t.Dict[str, str]]) -> bool:
        contents = [msg["content"] for msg in messages]
        classified = self.pipeline(contents)

        for res in classified:
            if res["label"] == "toxic":
                return False

        return True
