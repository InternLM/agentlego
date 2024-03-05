import heapq
import os
import re
from typing import Optional, Sequence
from urllib import parse

import requests

from agentlego.types import Annotated, Info
from agentlego.utils import require
from ..base import BaseTool
from ..utils.nlp import score_fasttext, score_naive, top_sentence


def extract_description(soup):
    description = soup.find(attrs={'name': 'description'})
    if description:
        content = description.get('content')
        if content:
            return content
    return None


def extract_snippet(query, url, ft, nlp, lang: str = 'en') -> Optional[str]:
    url = parse.unquote(url)
    response = requests.get(url=url, timeout=5)
    if response is None:
        return None
    response.encoding = 'utf-8'

    if lang == 'en':
        keywords = re.findall(r'\w+', query, re.ASCII)
    else:
        import jieba
        keywords = [
            word for word in jieba.cut(query) if word.strip() and word not in '，。！？—'
        ]

    # try to extract from page description
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    description = extract_description(soup)
    if description:
        if all(key_word in description for key_word in keywords):
            return description

    # extract related sentences from the webpage
    text = soup.get_text()
    sentences = re.split(r'\n|。|\.', text)
    scored_sentences = []
    for sentence in sentences:
        if 3 <= len(sentence) <= 200:
            scored_sentence = {
                'ft': -1 * score_fasttext(keywords, sentence, ft) if ft else None,
                'score_2': -1 * score_naive(keywords, sentence),
                'sentence': sentence,
            }
            scored_sentences.append(scored_sentence)

    top_sentences = heapq.nsmallest(
        5, scored_sentences, key=lambda x: x['ft'] or float('inf'))
    top_sentences += heapq.nsmallest(10, scored_sentences, key=lambda x: x['score_2'])

    stop_word = '。' if lang == 'zh' else '. '
    combined_text = stop_word.join([sentence['sentence'] for sentence in top_sentences])

    if len(combined_text) < 3:
        return None

    # Extract the top-3 related sentences
    try:
        summary = top_sentence(text=combined_text, limit=3, nlp=nlp)
        summary = ''.join(summary)
    except Exception:
        return None

    if any(keyword in summary for keyword in keywords):
        return summary
    else:
        return None


def filter_urls(urls,
                snippets,
                titles,
                black_list: Sequence[str] = ('youtube.com', 'bilibili.com', 'zhihu.com'),
                topk: int = 3):
    filtered_urls, filtered_snippets, filtered_titles = [], [], []
    count = 0
    for url, snippet, title in zip(urls, snippets, titles):
        if all(domain not in url
               for domain in black_list) and url.split('.')[-1] != 'pdf':
            filtered_urls.append(url)
            filtered_snippets.append(snippet)
            filtered_titles.append(title)
            count += 1
            if count >= topk:
                break

    return filtered_urls, filtered_snippets, filtered_titles


class BingSearch(BaseTool):
    default_desc = ('Search the input query text from Bing, '
                    'use it if you need need online information.')

    @require('en-core-web-sm', install='python -m spacy download en_core_web_sm')
    @require('zh-core-web-sm', install='python -m spacy download zh_core_web_sm')
    @require(['spacy', 'fasttext', 'langid', 'beautifulsoup4', 'jieba'])
    def __init__(self, sub_key: str = 'env', toolmeta=None):
        super().__init__(toolmeta=toolmeta)

        if sub_key == 'env':
            sub_key = os.getenv('BING_SUB_KEY')
        if not sub_key:
            raise ValueError('Please set Bing subscription key either in the environment'
                             ' as BING_SUB_KEY or pass it as `sub_key` parameter.')
        self.sub_key = sub_key

    def setup(self):
        import fasttext
        import spacy
        from bs4 import BeautifulSoup  # noqa: F401, F403
        from fasttext.util import download_model

        en_model = download_model('en', if_exists='ignore')
        zh_model = download_model('zh', if_exists='ignore')
        self.ft_en = fasttext.load_model(en_model)
        self.ft_zh = fasttext.load_model(zh_model)
        self.nlp_en = spacy.load('en_core_web_sm')
        self.nlp_zh = spacy.load('zh_core_web_sm')
        super().setup()

    def bing_search_api(self, query: str):
        endpoint = 'https://api.bing.microsoft.com/v7.0/search'
        params = {'q': query, 'mkt': 'zh-CN', 'count': '20'}
        headers = {'Ocp-Apim-Subscription-Key': self.sub_key}

        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    def apply(self,
              query: str,
              topk: Annotated[int, Info('The maximum number of results')] = 3) -> str:
        import langid

        langid.set_languages(['en', 'zh'])
        lang = langid.classify(query)[0]

        response = self.bing_search_api(query)

        webpages = {w['id']: w for w in response['webPages']['value']}
        raw_urls = []
        raw_snippets = []
        raw_titles = []
        for i, item in enumerate(response['rankingResponse']['mainline']['items']):
            if item['answerType'] == 'WebPages':
                webpage = webpages[item['value']['id']]
                if webpage is not None:
                    raw_urls.append(webpage['url'])
                    raw_snippets.append(webpage['snippet'])
                    raw_titles.append(webpage['name'])

            if item['answerType'] == 'News':
                if item['value']['id'] == response['news']['id']:
                    for n in response['news']['value']:
                        raw_urls.append(n['url'])
                        raw_snippets.append(n['description'])
                        raw_titles.append(n['name'])

        urls, snippets, titles = filter_urls(
            raw_urls, raw_snippets, raw_titles, topk=topk)

        docs = []
        for url, snippet, title in zip(urls, snippets, titles):
            try:
                ft = self.ft_zh if lang == 'zh' else self.ft_en
                nlp = self.nlp_zh if lang == 'zh' else self.nlp_en
                snippet = extract_snippet(
                    query, url, ft=ft, nlp=nlp, lang=lang) or snippet
            except Exception:
                pass
            summary = f'Title: {title}\nURL: {url}\n{snippet}'
            docs.append(summary)

        return '\n\n'.join(docs)
