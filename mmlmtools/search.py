# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from thefuzz import process

from mmlmtools.api import DEFAULT_TOOLS


def _cosine_similarity(a: np.array, b: np.array) -> list:
    """Calculate the cosine similarity of a and b."""
    dot_product = np.dot(b, a)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    res = dot_product / (norm_a * norm_b)
    return res


def _search_with_openai(query, choices, topk, model='text-embedding-ada-002'):
    try:
        from openai.embeddings_utils import get_embeddings
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'please install openai to enable searching tools powered by '
            'openai')

    embeddings = get_embeddings([query] + choices, engine=model)
    similarity = _cosine_similarity(
        np.array(embeddings[0]), np.array(embeddings[1:]))
    descend_sort = similarity.argsort()[::-1]

    return [choices[i] for i in descend_sort[:topk]]


def _serach_with_sentence_transformers(
        query, choices, topk, model='sentence-transformers/all-mpnet-base-v2'):
    model = SentenceTransformer(model)
    embeddings = model.encode([query] + choices)
    similarity = _cosine_similarity(embeddings[0], embeddings[1:])
    descend_sort = similarity.argsort()[::-1]

    return [choices[i] for i in descend_sort[:topk]]


def _search_with_thefuzz(query, choices, topk):
    result = process.extract(query, choices=choices, limit=topk)
    return [res for res, _ in result]


def search_tool(query: str, topk: int = 5, kind: str = 'thefuzz') -> List[str]:
    """Search several proper tools according to the query.

    Args:
        query (str): User input.
        topk (int): Number of tools to be returned.
        kind (str): Different third-party libraries are used to assist in
            searching the appropriate tools. Optional values are "thefuzz",
            "openai", and "st". Defaults to "thefuzz".

    Examples:
        >>> # use the thefuzz to search tools
        >>> search_tool('show the skeleton of person')
        >>> # use the openai API to search tools
        >>> search_tool('show the skeleton of person', kind='openai')
        >>> # use the sentence-transformers to search tools
        >>> search_tool('show the skeleton of person', kind='st')
    """
    choice2names = dict()
    for name, tool_meta in DEFAULT_TOOLS.items():
        choice2names[tool_meta['description']] = name

    choices = list(choice2names.keys())

    if kind == 'thefuzz':
        result = _search_with_thefuzz(query, choices, topk)
    elif kind == 'openai':
        result = _search_with_openai(query, choices, topk)
    elif kind == 'st':
        result = _serach_with_sentence_transformers(query, choices, topk)
    else:
        raise ValueError('The supported kind are "thefuzz", "openai" or "st", '
                         f'but got {kind}.')

    names = []
    for description in result:
        names.append(choice2names[description])

    return names
