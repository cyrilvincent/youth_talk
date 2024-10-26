import datetime
import difflib
import unicodedata
import os
from nltk.corpus import wordnet
from sqlentities import Lema, Topic, FormTopic
import spacy
import pytextrank

os.environ["NLTK_DATA"] = r"C:\Users\conta\git-CVC\Skema\git-youth_talk\youth_talk_1\nltk"


class NLTKModel:

    def __init__(self):
        self.min_length = 4
        self.synonym_dico: dict[str, list[str]] = {}
        self.topics: dict[str, Topic] = {}
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textrank")

    def textrank(self, text: str):
        doc = self.nlp(text)
        return [(phrase.text, phrase.count) for phrase in doc._.phrases]


    def gestalt(self, s1: str, s2: str):
        sm = difflib.SequenceMatcher(None, s1, s2)
        return sm.ratio()

    def gestalts(self, s: str, words: list[str]) -> tuple[str | None, float]:
        if s == "" or s is None:
            return None, 0
        max = 0
        res = None
        for item in words:
            if item != "":
                if s == item:
                    return item, 1
                if s.endswith("s"):
                    if s[:-1] == item:
                        return item, 0.99
                ratio = self.gestalt(s, item)
                if ratio > max:
                    max = ratio
                    res = item
                if max < 0.5 and s in item:
                    max = 0.5
        return res, max

    def get_synonyms(self, word: str) -> list[str]:
        if word in self.synonym_dico:
            return self.synonym_dico[word]
        synonyms = []
        for synonym in wordnet.synsets(word):
            for lema in synonym.lemmas():
                name = lema.name()
                if len(name) >= self.min_length and name not in synonyms:
                    synonyms.append(lema.name())
        self.synonym_dico[word] = synonyms
        return synonyms

    def nb_words(self, s: str) -> int:
        return 0 if s is None or len(s) == 0 else s.count(" ") + 1

    def sort_words(self, lemas: list[Lema]) -> list[Lema]:
        lemas = list(lemas)
        lemas.sort(key=lambda lema: len(lema.label))
        lemas.reverse()
        return lemas


    def strip_accents(self, s: str):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def remove_suffix(self, s: str):
        if s.endswith("ed"):
            return s[:-2]
        if s.endswith("ing"):
            return s[:-3]
        return s

    def normalize(self, s: str): # A porter dans lema
        s = s.lower()
        s = self.strip_accents(s)
        # s = self.remove_suffix(s)
        s = s.replace("(", "")
        s = s.replace(")", "")
        s = s.replace("'", "")
        s = s.replace('"', "")
        s = s.replace("-", " ")
        s = s.replace("+", " ")
        s = s.replace("/", " ")

        return s

    def split_phrases(self, text: str) -> list[str]:
        text = text.replace("\n", ".")
        text = text.replace("!", ".")
        text = text.replace(":", ".")
        text = text.replace("?", ".")
        text = text.replace(",", ".")
        return text.split(".")

    # def split_words(self, s: str) -> list[str]:
    #     return s.split(" ")

    def split_lemas(self, s: str) -> list[Lema]:
        words = s.split(" ")
        res = []
        previous = None
        for word in words:
            lema = Lema()
            lema.label, lema.previous = word, previous
            res.append(lema)
            previous = word
        return res
    #
    # def split_2_words(self, s: str) -> list[str]:
    #     previous: str | None = None
    #     words = self.split_words(s)
    #     pairs: list[str] = []
    #     for w in words:
    #         if w is not None and len(w) >= self.min_length and not self.is_link_word(w):
    #             if previous is not None:
    #                 pairs.append(f"{previous}_{w}")
    #             previous = w
    #         else:
    #             previous = None
    #     return pairs

    # def remove_short_words(self, words: list[str]) -> list[str]:
    #     return [w for w in words if len(w) >= self.min_length]

    def is_short(self, word: str) -> bool:
        if word == "war":
            return False
        return len(word) < self.min_length

    def remove_short_lemas(self, lemas: list[Lema]) -> list[Lema]:
        res = [lema for lema in lemas if not self.is_short(lema.label)]
        for lema in res:
            if lema.previous is not None and self.is_short(lema.previous):
                lema.previous = None
        return res

    def is_link_word(self, s: str) -> bool:
        if "@" in s:
            return True
        if s.isnumeric():
            return True
        links = ["with", "which", "when", "under", "upper", "ever", "never", "less", "more"]
        return s in links

    def remove_link_lemas(self, lemas: list[Lema]) -> list[Lema]:
        res = [lema for lema in lemas if not (self.is_link_word(lema.label))]
        for lema in res:
            if lema.previous is not None and self.is_link_word(lema.previous):
                lema.previous = None
        return res

    def get_sub_word47(self, word: str, lemas: list[Lema]) -> Lema | None:
        t = 4 / 7
        if word == "" or word is None:
            return None
        for lema in lemas:
            if word.startswith(lema.label):  # or lema.label.endswith(word):
                return lema
        return None

    def get_synonym(self, s: str, lemas: list[Lema]) -> Lema | None:
        synonyms = self.get_synonyms(s)
        if s in synonyms:
            synonyms.remove(s)
        if len(synonyms) == 0:
            return None
        for lema in lemas:
            if lema.label in synonyms:
                return lema
        return None

    def get_gestalt(self, s: str, lemas: list[Lema]) -> Lema | None:
        words = [lema.label for lema in lemas]
        word, score = self.gestalts(s, words)
        if score > 0.9:
            return [lema for lema in lemas if lema.label == word][0]
        return None

    def group_sub_word47(self):
        # for word in list(self.topics.keys()):
        #     if word in self.topics:
        #         topics = [topic for topic in self.topics.values() if topic.label != word]
        #         for topic in topics:
        #             lema = self.get_sub_word47(word, topic.lemas)
        #             if lema is not None:
        #                 self.topics[lema.label].count += self.topics[word].count
        #                 for lema2 in self.topics[word].lemas:
        #                     if lema2 not in self.topics[lema.label].lemas:
        #                         self.topics[lema.label].lemas.append(lema2)
        #                 del self.topics[word]
        self.group_generic(self.get_sub_word47)

    def group_generic(self, group_fn):
        for word in list(self.topics.keys()):
            if word in self.topics:
                topics = [topic for topic in self.topics.values() if topic.label != word]
                for topic in topics:
                    lema = group_fn(word, topic.lemas)
                    if lema is not None:
                        self.topics[word].count += self.topics[lema.label].count
                        for lema2 in list(self.topics[lema.label].lemas):
                            if lema2 not in self.topics[word].lemas:
                                self.topics[word].lemas.append(lema2)
                        del self.topics[lema.label]

    def group_synonyms(self):
        self.group_generic(self.get_synonym)

    def group_gestalts(self):
        self.group_generic(self.get_gestalt)

    def tokenize(self, text: str) -> list[list[Lema]]:
        text = self.normalize(text)
        phrases = self.split_phrases(text)
        lemass = [self.split_lemas(p) for p in phrases]
        lemass = [self.remove_short_lemas(lemas) for lemas in lemass]
        lemass = [self.remove_link_lemas(lemas) for lemas in lemass]
        lemass = [lemas for lemas in lemass if len(lemas) > 0]
        return lemass

    def count(self, phrases: list[list[Lema]]) -> dict[str, tuple[Lema, int]]:
        dico: dict[str, tuple[Lema, int]] = {}
        for phrase in phrases:
            for lema in phrase:
                if lema.label not in dico:
                    dico[lema.label] = lema, 0
                dico[lema.label] = lema, dico[lema.label][1] + 1
        return dico

    def grouping(self, dico: dict[str, tuple[Lema, int]]):
        for key in dico.keys():
            if key not in self.topics:
                topic = Topic()
                topic.label = key
                topic.count = dico[key][1]
                topic.lemas = [dico[key][0]]
                topic.source = "NLTK"
                topic.date = datetime.datetime.now()
                self.topics[key] = topic
        print(self.topics)
        self.group_sub_word47()
        print(self.topics)
        self.group_synonyms()
        print(self.topics)
        self.group_gestalts()
        print(self.topics)





