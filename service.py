import json
import typing as t
from typing import AsyncGenerator, Literal
from annotated_types import Ge, Le
from typing_extensions import Annotated
from enum import Enum, IntEnum
from pydantic import BaseModel, ConfigDict, ValidationError, Field

import bentoml
from bentoml.io import SSE
from bentoml.exceptions import BadInput
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

class ModelName(str, Enum):
    gpt3 = 'gpt-3.5-turbo'
    gpt4 = 'gpt-4o'
    mistral = 'mistral'


class GeneralChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    messages: t.List[t.Dict[str, str]]
    model: str
    stream: t.Optional[bool] = False


@bentoml.service(
    traffic={
        "concurrency": 100,
    },
    resources={
        "cpu": "8",
    },
)
class LLMGateway:

    def __init__(self):
        self.openai_client = AsyncOpenAI()

    @bentoml.api(input_spec=GeneralChatCompletionRequest, route="/v1/chat/completions")
    async def chat_completions(self, **params: t.Any) -> AsyncGenerator[str, None]:

        res = await self.openai_client.chat.completions.create(**params)
        async for chunk in res:
            yield SSE(json.dumps(chunk.dict())).marshal()
