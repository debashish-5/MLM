from pydantic import BaseModel, Field
from typing import Any


class TextIngestRequest(BaseModel):
    text:str = Field(min_length=1, max_length=20000)
    title:str = Field(default="Untitled",max_length=300)
    source:str = Field(default="user", max_length=500)
    page:int|None = None
    metadata:dict[str,any] = Field(default_factory=dict)


class QueryTextRequest(BaseModel):
    text:str | None = Field(default=None,max_length=5000)
    top_k:int = Field(default=5, ge = 1, le=20)
    include_ocr:bool = True

