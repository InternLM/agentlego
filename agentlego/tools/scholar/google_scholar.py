import os
from typing import Optional

from agentlego.types import Annotated, Info
from agentlego.utils import require
from ..base import BaseTool


class GoogleScholarArticle(BaseTool):
    default_desc = ('Search for scholarly articles based on'
                    ' a query according to the google scholar.')

    @require('google-search-results')
    def __init__(self, api_key: str = 'env', timeout: int = 5, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        if api_key == 'env':
            api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            raise ValueError('Please set SerpAPI API key either in the environment'
                             ' as SERPAPI_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout

    def apply(
        self,
        query: str,
        as_ylo: Annotated[Optional[int],
                          Info('The starting year for results')] = None,
        as_yhi: Annotated[Optional[int],
                          Info('The ending year for results')] = None,
        num: Annotated[int, Info('The maximum number of results, limited to 20')] = 3,
    ) -> Annotated[str,
                   Info('Article information, include title, '
                        'organic id, publication and snippets')]:
        from serpapi import GoogleSearch
        params = dict(
            q=query,
            engine='google_scholar',
            api_key=self.api_key,
            as_ylo=as_ylo,
            as_yhi=as_yhi,
            num=num,
        )
        search = GoogleSearch(params)
        results = search.get_dict()['organic_results'][:num]
        docs = []
        for item in results:
            citation = item.get('inline_links', {}).get('cited_by', {}).get('total', '')
            publication = item.get('publication_info', {}).get('summary', '')
            docs.append(f"Title: {item.get('title', '')}\n"
                        f'Publication: {publication}\n'
                        f"Organic-id: {item.get('result_id', '')}\n"
                        f"Snippet: {item.get('snippet', '')}\n"
                        f'Total-Citations: {citation}')
        return '\n\n'.join(docs)


class GoogleScholarAuthorInfo(BaseTool):
    default_desc = "Search for an author's information by author's id."

    @require('google-search-results')
    def __init__(self, api_key: str = 'env', timeout: int = 5, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        if api_key == 'env':
            api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            raise ValueError('Please set SerpAPI API key either in the environment'
                             ' as SERPAPI_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout

    def apply(self, author_id: Annotated[str, Info('ID of the author')]) -> str:
        from serpapi import GoogleSearch
        params = dict(
            engine='google_scholar_author',
            api_key=self.api_key,
            author_id=author_id,
        )
        search = GoogleSearch(params)
        results = search.get_dict()
        author = results.get('author')
        if not author:
            return 'No author is found, please check your author id.'

        articles = results.get('articles', [])
        docs = [
            f"Name: {author.get('name', '')}",
            f"Affiliations: {author.get('affiliations', '')}",
            f"Website: {author.get('website', '')}",
        ]
        if articles:
            docs.append('Articles: ')
            for article in articles[:3]:
                cite = article.get('cited_by', {}).get('value')
                cite = f' ({cite} citations)' if cite is not None else ''
                docs.append(f"- {article['title']}{cite}")
        return '\n'.join(docs)


class GoogleScholarAuthorId(BaseTool):
    default_desc = "Get the author's id by name."

    @require('google-search-results')
    def __init__(self, api_key: str = 'env', timeout: int = 5, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        if api_key == 'env':
            api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            raise ValueError('Please set SerpAPI API key either in the environment '
                             ' as SERPAPI_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout

    def apply(
        self, query: Annotated[str,
                               Info('Author name or other related information')]
    ) -> Annotated[str, Info('The author id of the author')]:
        from serpapi import GoogleSearch
        params = dict(
            mauthors=query,
            engine='google_scholar_profiles',
            api_key=self.api_key,
        )
        search = GoogleSearch(params)
        results = search.get_dict()
        profile = results.get('profiles', [])
        if not profile:
            return 'No author is found.'
        return profile[0]['author_id']


class GoogleScholarCitation(BaseTool):
    default_desc = 'Get the citation text in all styles of an article by organic id.'

    @require('google-search-results')
    def __init__(self, api_key: str = 'env', timeout: int = 5, toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        if api_key == 'env':
            api_key = os.getenv('SERPAPI_API_KEY')
        if not api_key:
            raise ValueError('Please set SerpAPI API key either in the environment '
                             ' as SERPAPI_API_KEY or pass it as `api_key` parameter.')
        self.api_key = api_key
        self.timeout = timeout

    def apply(self, organic_id: Annotated[str,
                                          Info('The organic id of an article')]) -> str:
        from serpapi import GoogleSearch
        params = dict(
            q=organic_id,
            engine='google_scholar_cite',
            api_key=self.api_key,
        )
        search = GoogleSearch(params)
        results = search.get_dict()
        citations = results['citations']
        docs = []
        for citation in citations:
            docs.append(f"{citation['title']}: {citation['snippet']}")
        return '\n'.join(docs)
