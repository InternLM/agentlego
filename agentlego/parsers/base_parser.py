from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from agentlego.schema import ToolMeta
    from agentlego.tools.base import BaseTool


class BaseParser(metaclass=ABCMeta):

    def __init__(self, tool: 'BaseTool') -> None:
        self.tool: 'BaseTool' = tool

    @property
    def toolmeta(self) -> 'ToolMeta':
        return self.tool.toolmeta

    @abstractmethod
    def parse_inputs(self, *args, **kwargs) -> Tuple[Tuple, Dict]:
        raise NotImplementedError

    @abstractmethod
    def parse_outputs(self, outputs: Any) -> Any:
        raise NotImplementedError

    def refine_description(self) -> str:
        return self.toolmeta.description
