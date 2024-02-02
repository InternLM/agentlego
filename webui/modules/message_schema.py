from typing import Literal, Mapping, Optional, Tuple

from pydantic import BaseModel


class Message(BaseModel):
    ...


class ToolInput(Message):
    name: str
    args: Mapping[str, dict]
    thought: Optional[str] = None
    _role: Literal['ToolInput'] = 'ToolInput'


class ToolOutput(Message):
    outputs: Optional[Tuple[dict, ...]] = None
    error: Optional[str] = None
    _role: Literal['ToolOutput'] = 'ToolOutput'


class Answer(Message):
    text: str
    thought: Optional[str] = None
    _role: Literal['Answer'] = 'Answer'


class Error(Message):
    type: str
    reason: Optional[str] = None
    _role: Literal['Answer'] = 'Answer'
