import io

from func_timeout import func_set_timeout
from PIL import Image

from agentlego.types import Annotated, ImageIO, Info
from agentlego.utils import require
from ..base import BaseTool
from .python_interpreter import GenericRuntime

DESC_EN = '''\
This tool can execute Python code to plot diagrams. The code should include a function named 'solution'. The function should return the matplotlib figure directly. Avoid printing the answer. The code instance format is as follows:

```python
# import packages
import matplotlib.pyplot as plt
def solution():
    # labels and data
    cars = ['AUDI', 'BMW', 'FORD', 'TESLA', 'JAGUAR', 'MERCEDES']
    data = [23, 17, 35, 29, 12, 41]

    # draw diagrams
    figure = plt.figure(figsize=(8, 6))
    plt.pie(data, labels=cars, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Car Distribution')
    return figure
```
'''  # noqa: E501


class Plot(BaseTool):
    """A tool to plot diagrams using python interpreter.

    WARNING: The PythonInterpreter only has minimal protection, don't expose to
    trustless environment.

    Args:
        timeout (int, Optional): Upper bound of waiting time for Python script execution.
            Defaults to ``20``.
        toolmeta (None | dict | ToolMeta): The additional info of the tool.
            Defaults to None.
    """

    default_desc = DESC_EN
    answer_expr = 'solution()'

    @require('matplotlib')
    def __init__(self, timeout: int = 20, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.timeout = timeout

    def apply(self, command: Annotated[str,
                                       Info('Markdown format Python code')]) -> ImageIO:

        if '```python' in command:
            command = command.split('```python')[1].split('```')[0]
        elif '```' in command:
            command = command.split('```')[1].split('```')[0]

        res = func_set_timeout(self.timeout)(self._call)(command)
        assert res is not None
        return ImageIO(res)

    def _call(self, command: str) -> Image.Image:
        from matplotlib.pyplot import Figure

        runtime = GenericRuntime(headers=['import matplotlib.pyplot as plt'])
        runtime.exec_code('\n'.join(command))
        figure: Figure = runtime.eval_code(self.answer_expr)
        if not isinstance(figure, Figure):
            raise TypeError('The `solution` function must return the matplotlib figure.')
        buf = io.BytesIO()
        figure.savefig(buf)
        buf.seek(0)
        return Image.open(buf)
