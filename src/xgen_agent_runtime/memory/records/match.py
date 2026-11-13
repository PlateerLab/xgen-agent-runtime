"""Lexical relevance between a context (query, recent turns) and records — no model, no network.

Used where a decision must not depend on an embedding being configured: which rules apply now,
which facts relate to this turn, which existing records a new candidate may be the same as.
A vector index, when present, is added on top (``assemble`` / ``reconcile``), never required.

Korean is written with particles glued to words ("보고서를", "보고서는"), so words alone miss;
character bigrams of Hangul runs catch them. Latin/digits are compared as words.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = ["terms", "Scorer"]

_WORD_RE = re.compile(r"[0-9a-z]+(?:[._-][0-9a-z]+)*|[가-힣]+")
#: Very common words that would make everything look related.
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "is",
    "are",
    "be",
    "it",
    "this",
    "that",
    "with",
    "as",
    "at",
    "by",
    "we",
    "you",
    "i",
    "그리고",
    "하는",
    "있는",
    "있다",
    "한다",
    "합니다",
    "입니다",
    "그",
    "이",
    "저",
    "것",
    "수",
    "등",
    "및",
}


def terms(text: str) -> List[str]:
    """Comparison terms: lower-cased words (Latin/digits), Hangul words and their bigrams."""
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    out: List[str] = []
    for w in _WORD_RE.findall(t):
        if w in _STOP:
            continue
        if "가" <= w[0] <= "힣":
            if len(w) >= 2:
                out.append(w)
                out.extend(w[i : i + 2] for i in range(len(w) - 1))
        elif len(w) >= 2 or w.isdigit():
            out.append(w)
    return out


#: ``rank`` skips a term found in more than this share of the documents (and more than ``_COMMON_MIN``).
_COMMON_SHARE = 0.2
_COMMON_MIN = 100


class Scorer:
    """BM25-style overlap over a fixed set of documents (the records being ranked).

    Rare shared terms count for more than common ones, long records do not win by length.
    """

    def __init__(self, docs: Sequence[str], *, k1: float = 1.2, b: float = 0.6) -> None:
        self._docs = [Counter(terms(d)) for d in docs]
        self._len = [sum(c.values()) or 1 for c in self._docs]
        self._avg = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Dict[str, int] = {}
        for c in self._docs:
            for term in c:
                df[term] = df.get(term, 0) + 1
        n = max(1, len(self._docs))
        self._idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self._k1, self._b = k1, b
        # Postings: ranking touches only documents that share a term with the query.
        self._postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, c in enumerate(self._docs):
            for term, f in c.items():
                self._postings.setdefault(term, []).append((i, f))

    def score(self, i: int, query_terms: Iterable[str]) -> float:
        doc = self._docs[i]
        if not doc:
            return 0.0
        norm = self._k1 * (1 - self._b + self._b * self._len[i] / self._avg)
        total = 0.0
        for term in set(query_terms):
            f = doc.get(term)
            if f:
                total += self._idf.get(term, 0.0) * f * (self._k1 + 1) / (f + norm)
        return total

    def rank(self, query: str, *, min_score: float = 0.0) -> List[Tuple[int, float]]:
        totals: Dict[int, float] = {}
        # A term in a large share of a large memory says little and costs a pass over most of it.
        common = max(_COMMON_MIN, int(len(self._docs) * _COMMON_SHARE))
        for term in set(terms(query)):
            idf = self._idf.get(term)
            if not idf:
                continue
            postings = self._postings.get(term, ())
            if len(postings) > common:
                continue
            for i, f in postings:
                norm = self._k1 * (1 - self._b + self._b * self._len[i] / self._avg)
                totals[i] = totals.get(i, 0.0) + idf * f * (self._k1 + 1) / (f + norm)
        return sorted(((i, sc) for i, sc in totals.items() if sc > min_score), key=lambda x: -x[1])


def coverage(query: str, phrase: str) -> float:
    """How much of ``phrase`` (e.g. a rule's "applies when") the query covers, 0..1."""
    p = set(terms(phrase))
    if not p:
        return 0.0
    q = set(terms(query))
    return len(p & q) / len(p)
