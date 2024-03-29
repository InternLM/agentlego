from collections import Counter
from string import punctuation
from typing import List

import numpy as np


def top_sentence(text: str, limit: int, nlp) -> List[str]:
    # Copyright 2023 piglake
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.

    keyword = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
    doc = nlp(text.lower())
    for token in doc:
        if (token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            keyword.append(token.text)

    freq_word = Counter(keyword)
    max_freq = Counter(keyword).most_common(1)[0][1]
    for w in freq_word:
        freq_word[w] = (freq_word[w] / max_freq)
    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.text in freq_word.keys():
                if sent in sent_strength.keys():
                    sent_strength[sent] += freq_word[word.text]
                else:
                    sent_strength[sent] = freq_word[word.text]
    summary = []

    sorted_x = sorted(sent_strength.items(), key=lambda kv: kv[1], reverse=True)

    counter = 0
    for i in range(len(sorted_x)):
        summary.append(str(sorted_x[i][0]).capitalize())
        counter += 1
        if (counter >= limit):
            break

    return summary


def score_fasttext(keywords: List[str], sentence: str, ft) -> float:
    res = 0

    for keyword in keywords:
        key_embedding = ft.get_word_vector(keyword)
        vector = ft.get_word_vector(sentence)  # 300-dim vector
        cos_sim = np.dot(key_embedding, vector) / (
            np.linalg.norm(key_embedding) * np.linalg.norm(vector))
        res += cos_sim
    return res


def score_naive(keywords: List[str], sentence: str) -> float:
    return float(sum((keyword in sentence) for keyword in keywords))
