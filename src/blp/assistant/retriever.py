"""Hybrid retrieval ภาษาไทย (บทที่ 6.2): BM25 ที่ตัดคำด้วย PyThaiNLP + TF-IDF character n-gram
รวมอันดับด้วย reciprocal rank fusion (RRF) และกรองด้วย metadata (ภาค ไตรมาส สาขา)
งานจริงเปลี่ยน TF-IDF เป็น dense embedding (เช่น BGE-M3) ได้ผ่าน embed_fn โดยไม่แก้ส่วนอื่น
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def thai_tokenize(text: str) -> list[str]:
    try:
        from pythainlp.tokenize import word_tokenize
        return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]
    except ImportError:                                      # สำรอง: character bigram
        s = "".join(text.split())
        return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.cv = CountVectorizer(tokenizer=thai_tokenize, lowercase=False, token_pattern=None)
        self.tf = self.cv.fit_transform(texts).tocsc().astype(float)
        dl = np.asarray(self.tf.sum(axis=1)).ravel()
        self.norm = k1 * (1 - b + b * dl / max(dl.mean(), 1e-9))
        df = np.diff(self.tf.indptr)
        n = self.tf.shape[0]
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        self.k1 = k1

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(self.tf.shape[0])
        for j in self.cv.transform([query]).indices:
            lo, hi = self.tf.indptr[j], self.tf.indptr[j + 1]
            rows, tf = self.tf.indices[lo:hi], self.tf.data[lo:hi]
            s[rows] += self.idf[j] * tf * (self.k1 + 1) / (tf + self.norm[rows])
        return s


def _ranks(scores: np.ndarray) -> np.ndarray:
    r = np.empty(len(scores))
    r[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return r


class HybridRetriever:
    def __init__(self, docs: pd.DataFrame, text_col: str = "text_body",
                 embed_fn: Callable[[list[str]], np.ndarray] | None = None):
        self.docs = docs.reset_index(drop=True)
        texts = self.docs[text_col].tolist()
        self.bm25 = BM25(texts)
        if embed_fn is None:
            vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), sublinear_tf=True)
            X = vec.fit_transform(texts)
            self._dense = lambda q: np.asarray((X @ vec.transform([q]).T).todense()).ravel()
        else:
            E = np.asarray(embed_fn(texts), dtype=float)
            E /= np.linalg.norm(E, axis=1, keepdims=True)

            def dense(q: str) -> np.ndarray:
                v = np.asarray(embed_fn([q]), dtype=float)[0]
                return E @ (v / np.linalg.norm(v))
            self._dense = dense

    def search(self, query: str, k: int = 8, filters: dict | None = None, rrf_k: int = 60) -> pd.DataFrame:
        mask = np.ones(len(self.docs), dtype=bool)
        for col, val in (filters or {}).items():
            mask &= (self.docs[col].astype(str) == str(val)).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return self.docs.iloc[[]]
        rrf = 1 / (rrf_k + _ranks(self.bm25.scores(query)[idx])) + 1 / (rrf_k + _ranks(self._dense(query)[idx]))
        order = np.argsort(-rrf, kind="stable")[:k]
        return self.docs.iloc[idx[order]].assign(score_rrf=rrf[order])
