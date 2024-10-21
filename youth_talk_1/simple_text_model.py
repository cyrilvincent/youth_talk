import difflib
import unicodedata
from typing import LiteralString


class SimpleTextModel:

    def __init__(self):
        self.min_length = 4

    def gestalt(self, s1: str, s2: str):
        sm = difflib.SequenceMatcher(None, s1, s2)
        return sm.ratio()

    def gestalts(self, s: str, l: list[str]) -> tuple[str | None, float]:
        if s == "" or s is None:
            return None, 0
        max = 0
        res = None
        for item in l:
            if item != "":
                if s == item:
                    return item, 1
                ratio = self.gestalt(s, item)
                if ratio > max:
                    max = ratio
                    res = item
                if max < 0.5 and s in item:
                    max = 0.5
        return res, max

    def is_sub_word(self, s: str, l: list[str]) -> str | None:
        if s == "" or s is None:
            return None
        for item in l:
            if item != "":
                if (s.startswith(item) or s.endswith(item) or item.startswith(s) or item.endswith(s))\
                        and (" " in s or " " in item):
                    return item
        return None

    def strip_accents(self, s: str):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def remove_suffix(self, s: str):
        if s.endswith("ed"):
            return s[:-2]
        if s.endswith("ing"):
            return s[:-3]
        return s

    def normalize(self, s: str):
        s = s.lower()
        s = self.strip_accents(s)
        s = self.remove_suffix(s)
        s = s.replace("(", "")
        s = s.replace(")", "")
        s = s.replace("'", "")
        s = s.replace('"', "")
        s = s.replace("-", " ")
        s = s.replace("+", " ")
        s = s.replace("/", " ")


        return s

    def split_phrases(self, s: str) -> list[str]:
        s = s.replace("\n", ".")
        s = s.replace("!", ".")
        s = s.replace(":", ".")
        s = s.replace("?", ".")
        s = s.replace(",", ".")
        return s.split(".")

    def split_words(self, s: str) -> list[str]:
        return s.split(" ")

    def split_2_words(self, s: str) -> list[str]:
        previous: str | None = None
        l = self.split_words(s)
        pairs: list[str] = []
        for w in l:
            if w is not None and len(w) >= self.min_length and not self.is_link_words(w):
                if previous is not None:
                    pairs.append(f"{previous} {w}")
                previous = w
            else:
                previous = None
        return pairs

    def remove_short_words(self, l: list[str]) -> list[str]:
        return [w for w in l if len(w) >= self.min_length]

    def is_link_words(self, s: str) -> bool:
        return s == "with" or s == "which" or s == "when" or s == "under" or "@" in s

    def remove_link_words(self, l: list[str]) -> list[str]:
        return [w for w in l if not(self.is_link_words(w))]

    def count(self, phrases: list[list[str]]):
        dico: dict[str, int] = {}
        for p in phrases:
            for w in p:
                if w not in dico:
                    dico[w] = 0
                dico[w] += 1
        return dico

    def tokenize(self, s: str) -> list[list[str]]:
        s = self.normalize(s)
        l = self.split_phrases(s)
        phrases = [self.split_words(p) for p in l]
        phrases = [self.remove_short_words(words) for words in phrases]
        phrases = [self.remove_link_words(words) for words in phrases]
        two_words = [self.split_2_words(p) for p in l]
        phrases = [p + t for p, t in zip(phrases, two_words)]
        phrases = [words for words in phrases if len(words) > 0]
        return phrases

    def grouping(self, dico:dict[str, str]) -> dict[str, tuple[list[str], int]]:
        res: dict[str, tuple[list[str], int]] = {}
        for w in dico.keys():
            if w in res:
                res[w][1] += 1
            else:
                word = self.is_sub_word(w, list(res.keys()))
                if word is not None:
                    if w not in res[word][0]:
                        res[word][0].append(w)
                else:
                    word, score = self.gestalts(w, list(res.keys()))
                    if score > 0.9:
                        res[word] = res[word][0], res[word][1] + 1
                        if w not in res[word][0]:
                            res[word][0].append(w)
                    else:
                        res[w] = [], 1
        return res

    # form *-* topic -* word
    # form = formulaire
    # topic = unique(form_id, topic), nb
    # topic.topic = key du dict de grouping topic.nb = int duc dict
    # word = word = str list[str] du dico






