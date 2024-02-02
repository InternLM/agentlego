from typing import List

import numpy as np

from .apis.tool import list_tools
from .utils import load_or_build_object


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate the cosine similarity of a and b."""
    dot_product = np.dot(b, a)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    res = dot_product / (norm_a * norm_b)
    return res


def _search_with_openai(query, choices, model='text-embedding-ada-002', topk=5):
    """Search tools with openai API.

    Note:
        You need to install openai first. And you need to set the
        OPENAI_API_KEY.

    Args:
        query (str): User input.
        choices (list): List of tool descriptions.
        topk (int): Max number of tools to be returned.
        model (str): OpenAI API model name.
            Defaults to 'text-embedding-ada-002'.

    Returns:
        list: List of tool descriptions.
    """
    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            'please install openai to enable searching tools powered by '
            'openai')

    client = OpenAI()
    embeddings = client.embeddings.create(input=[query] + choices, model=model).data
    similarity = _cosine_similarity(np.array(embeddings[0]), np.array(embeddings[1:]))

    indices = np.argsort(-similarity)[:topk]
    return [choices[i] for i in indices]


def _serach_with_sentence_transformers(query,
                                       choices,
                                       model='sentence-transformers/all-mpnet-base-v2',
                                       topk=5):
    """Search tools with sentence-transformers.

    Args:
        query (str): User input.
        choices (list): List of tool descriptions.
        model (str): Sentence-transformers model name. Defaults to
            'sentence-transformers/all-mpnet-base-v2'.

    Returns:
        list: List of tool descriptions.
    """
    from sentence_transformers import SentenceTransformer

    model = load_or_build_object(SentenceTransformer, model)
    embeddings = model.encode([query] + choices)
    similarity = _cosine_similarity(embeddings[0], embeddings[1:])

    indices = np.argsort(-similarity)[:topk]
    return [choices[i] for i in indices]


def _search_with_thefuzz(query, choices, topk=5):
    from thefuzz import process
    result = process.extract(query, choices=choices, limit=topk)
    return [res for res, _ in result]


def search_tool(query: str, kind: str = 'thefuzz', topk=5) -> List[str]:
    """Search several proper tools according to the query.

    Args:
        query (str): User input.
        kind (str): Different third-party libraries are used to assist in
            searching the appropriate tools. Optional values are "thefuzz",
            "openai", and "st". Defaults to "thefuzz".
        topk (int): Return the top-k results. Defaults to 5.

    Examples:
        >>> from agentlego import search_tool
        >>> # use the thefuzz to search tools
        >>> search_tool('human pose')
        >>> # use the openai API to search tools
        >>> search_tool('human pose', kind='openai')
        >>> # use the sentence-transformers to search tools
        >>> search_tool('human pose', kind='st')
    """
    choice2names = dict()
    for name, description in list_tools(with_description=True):
        choice2names[description] = name

    choices = list(choice2names.keys())

    if kind == 'thefuzz':
        result = _search_with_thefuzz(query, choices, topk=topk)
    elif kind == 'openai':
        result = _search_with_openai(query, choices, topk=topk)
    elif kind == 'st':
        result = _serach_with_sentence_transformers(query, choices, topk=topk)
    else:
        raise ValueError('The supported kinds are "thefuzz", "openai" or "st",'
                         f' but got {kind}.')

    names = []
    for description in result:
        names.append(choice2names[description])

    return names
