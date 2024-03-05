from agentlego.utils import require
from ..base import BaseTool


class ArxivSearch(BaseTool):
    default_desc = 'Run Arxiv search and get the article meta information.'

    @require('arxiv')
    def __init__(self,
                 top_k_results: int = 3,
                 max_query_len: int = 300,
                 doc_content_chars_max: int = 1500,
                 toolmeta=None):
        super().__init__(toolmeta=toolmeta)
        self.top_k_results = top_k_results
        self.max_query_len = max_query_len
        self.doc_content_chars_max = doc_content_chars_max

    def apply(self, query: str) -> str:
        import arxiv

        results = arxiv.Search(
            query[:self.max_query_len], max_results=self.top_k_results).results()

        if not results:
            return 'No good Arxiv Result was found'

        docs = []
        for result in results:
            summary = result.summary
            if len(summary) > self.doc_content_chars_max:
                summary = summary[:self.doc_content_chars_max] + '...'
            docs.append(f'Published: {result.updated.date()}\n'
                        f'Title: {result.title}\n'
                        f'Authors: {", ".join(a.name for a in result.authors)}\n'
                        f'Summary: {summary}')
        return '\n\n'.join(docs)
