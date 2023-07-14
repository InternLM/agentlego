# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

from thefuzz import process

from mmlmtools.api import DEFAULT_TOOLS


def search_tool(query: str, topk: int = 2) -> List[str]:
    """Search several proper tools according to the query.

    Args:
        query (str): User input.
        topk (int): Number of tools to be returned.
    """
    choice2names = dict()
    for name, tool_meta in DEFAULT_TOOLS.items():
        choice2names[tool_meta['description']] = name

    choices = list(choice2names.keys())
    result = process.extract(query, choices=choices, limit=topk)
    names = []
    for description, _ in result:
        names.append(choice2names[description])

    return names
