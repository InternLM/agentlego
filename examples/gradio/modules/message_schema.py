from typing import Any, Literal, Optional, Tuple, Union

from pydantic import BaseModel


class ToolInput(BaseModel):
    name: str
    args: Union[str, dict, None] = None
    thought: Optional[str] = None
    _role: Literal['ToolInput'] = 'ToolInput'


class ToolOutput(BaseModel):
    outputs: Optional[Tuple[Any, ...]] = None
    error: Optional[str] = None
    _role: Literal['ToolOutput'] = 'ToolOutput'


class Answer(BaseModel):
    text: str
    thought: Optional[str] = None
    _role: Literal['Answer'] = 'Answer'

class Error(BaseModel):
    type: str
    reason: Optional[str] = None
    _role: Literal['Answer'] = 'Answer'
