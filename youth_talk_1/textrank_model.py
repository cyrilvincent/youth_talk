import difflib
import unicodedata
from nltk.corpus import wordnet
from sqlalchemy import select

from dbcontext import Context
from sqlentities import Lema, Topic, Form, Stat
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import pytextrank
import config
import art
import argparse


class TextrankModel:

    def __init__(self):
        self.min_length = 4
        self.gestalt_ratio = 0.9
        self.synonym_dico: dict[str, list[str]] = {}
        self.topics: dict[str, Topic] = {}
        self.nlp = spacy.load("en_core_web_sm")  # python -m spacy download en_core_web_sm
        self.nlp.add_pipe("textrank")
        self.sentiment_analyser = SentimentIntensityAnalyzer()

    def textrank(self, text: str) -> list[tuple[str, int]]:
        doc = self.nlp(text)
        return [(phrase.text, phrase.count) for phrase in doc._.phrases]

    def sentiment(self, text: str) -> dict[str, float]:
        return self.sentiment_analyser.polarity_scores(text)["compound"]

    def nb_words(self, text: str) -> int:
        if len(text.strip()) == 0:
            return 0
        text = text.replace("\n", " ")
        text = text.replace("|", " ")
        return text.count(" ") + 1


    def gestalt(self, s1: str, s2: str):
        sm = difflib.SequenceMatcher(None, s1, s2)
        return sm.ratio()

    def gestalts(self, s: str, words: list[str]) -> tuple[str | None, float]:
        if s == "" or s is None:
            return None, 0
        max = 0
        res = None
        if s == "mariage":
            pass
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

    def strip_accents(self, s: str):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def normalize(self, s: str):
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
        text = text.replace("|", ".")
        text = text.replace("\n", ".")
        text = text.replace("!", ".")
        text = text.replace(":", ".")
        text = text.replace("?", ".")
        text = text.replace(",", ".")
        return text.split(".")

    def split_words(self, s: str) -> list[str]:
        return s.split(" ")

    def split_lemas(self, s: str) -> list[Lema]:
        words = s.split(" ")
        res = []
        previous = None
        pre_previous = None
        for word in words:
            word = word.strip()
            if len(word) > 0:
                lema = Lema(word, previous, pre_previous)
                res.append(lema)
                pre_previous = previous
                previous = word
        return res

    def is_short(self, word: str) -> bool:
        if word == "war":
            return False
        return len(word) < self.min_length

    def remove_short_lemas(self, lemas: list[Lema]) -> list[Lema]:
        res = [lema for lema in lemas if not self.is_short(lema.label)]
        for lema in res:
            if lema.previous is not None and self.is_short(lema.previous):
                lema.previous = None
            if lema.pre_previous is not None and self.is_short(lema.pre_previous):
                lema.pre_previous = None
        return res

    def is_link_word(self, s: str) -> bool:
        if "@" in s:
            return True
        if s.isnumeric():
            return True
        links = ["with", "which", "when", "under", "upper", "ever", "never", "less", "more", "true", "false"]
        return s in links

    def remove_link_lemas(self, lemas: list[Lema]) -> list[Lema]:
        res = [lema for lema in lemas if not (self.is_link_word(lema.label))]
        return res

    def get_sub_word47(self, word: str, lemas: list[Lema]) -> Lema | None:
        t = 4 / 7
        if word == "" or word is None:
            return None
        if word == "mariage":
            pass
        for lema in lemas:
            w1 = word
            if w1.endswith("ing"):
                w1 = w1[:-3]
            elif w1.endswith("ed"):
                w1 = w1[:2]
            w2 = lema.label
            if w2.endswith("ing"):
                w2 = w2[:-3]
            elif w2.endswith("ed"):
                w2 = w2[:2]
            if (w1.startswith(w2) or w2.startswith(w1)) and len(w1) / len(w2) > t:
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
        if score > self.gestalt_ratio:
            return [lema for lema in lemas if lema.label == word][0]
        return None

    def group_sub_word47(self):
        self.group_generic(self.get_sub_word47)

    def group_generic(self, group_fn):
        cloned = dict(self.topics)
        for word in cloned.keys():
            if word in self.topics:
                topics = [topic for topic in self.topics.values() if topic.label != word]
                for topic in topics:
                    if word == "mariage":
                        pass
                    lema = group_fn(word, topic.lemas)
                    if lema is not None:
                        self.topics[word].count += self.topics[topic.label].count
                        for lema2 in list(self.topics[topic.label].lemas):
                            if lema2 not in self.topics[word].lemas:
                                self.topics[word].lemas.append(lema2)
                        del self.topics[topic.label]

    def group_synonyms(self):
        self.group_generic(self.get_synonym)

    def group_gestalts(self):
        self.group_generic(self.get_gestalt)

    # Utiliser tokenize_textrank à la place
    def tokenize(self, text: str) -> list[list[Lema]]:
        text = self.normalize(text)
        phrases = self.split_phrases(text)
        lemass = [self.split_lemas(p) for p in phrases]
        return self.tokenize_lemass(lemass)

    def tokenize_lemass(self, lemass: list[list[Lema]]) -> list[list[Lema]]:
        lemass = [self.remove_short_lemas(lemas) for lemas in lemass]
        lemass = [self.remove_link_lemas(lemas) for lemas in lemass]
        lemass = [lemas for lemas in lemass if len(lemas) > 0]
        return lemass

    def tokenize_textrank(self, text: str, limit=config.textrank_limit) -> list[list[Lema]]:
        text = self.normalize(text)
        phrases = self.textrank(text)
        lemass = [self.split_lemas(p[0]) * p[1] for p in phrases[:limit]]
        lemass = self.tokenize_lemass(lemass)
        return lemass

    def count(self, phrases: list[list[Lema]]) -> dict[str, Lema]:
        dico: dict[str, Lema] = {}
        for phrase in phrases:
            for lema in phrase:
                if lema.label not in dico:
                    dico[lema.label] = lema
                dico[lema.label].count += 1
        return dico

    def grouping(self, dico: dict[str, Lema]):
        for key in dico.keys():
            if key not in self.topics:
                topic = Topic(label=key, source="textrank", lemas=[dico[key]])
                self.topics[key] = topic
            self.topics[key].count = dico[key].count # Ca bugera avec une bd, mais je ne suis vraiment pas sur que topic.count serve car qu'il soit présent 2 fois ou 1 fois pour moi c'est idem
        print(self.topics)
        self.group_sub_word47()
        print(self.topics)
        self.group_synonyms()
        print(self.topics)
        self.group_gestalts()
        print(self.topics)

    def doubling(self):
        for key in self.topics.keys():
            cloned = list(self.topics[key].lemas)
            for lema in cloned:
                if lema.previous is not None:
                    lema2 = Lema(f"{lema.previous}_{lema.label}")
                    if lema2 not in self.topics[key].lemas:
                        self.topics[key].lemas.append(lema2)
                    lema2 = [lema for lema in self.topics[key].lemas if lema == lema2][0]
                    lema2.count += 1
                    if lema.pre_previous is not None:
                        lema3 = Lema(f"{lema.pre_previous}_{lema.previous}_{lema.label}")
                        if lema3 not in self.topics[key].lemas:
                            self.topics[key].lemas.append(lema3)
                        lema3 = [lema for lema in self.topics[key].lemas if lema == lema3][0]
                        lema3.count += 1


class TextrankService:

    def __init__(self, context):
        self.context = context
        self.model = TextrankModel()
        self.nb_form = 0

    def average(self, *args) -> float | None:
        nb = 0
        sum = 0
        for arg in args:
            if arg is not None:
                nb += 1
                sum += arg
        if nb == 0:
            return None
        return sum / nb

    def categorize(self, score: float | None) -> int | None:
        if score is None:
            return None
        if score < 2.33:
            return 0
        if score > 3.67:
            return 2
        return 1

    def analyse(self):
        # joinload = self.session.execute(select(entities.Cart, 1).options(joinedload(entities.Cart.medias))).scalars().first()
        # join + joinload = session.execute(select(Media).options(joinedload(Media.publisher)).join(Publisher).where(Publisher.name == "Vincent"))
        forms: list[Form] = self.context.session.execute(select(Form).where(Form.empathy_answers > 0)).scalars().all()
        for form in forms:
            if (form.question_01_contrib1_answer is not None
                    or form.question_01_contrib2_answer is not None
                    or form.question_01_contrib3_answer is not None):
                self.analyse_form(form)
        print("Committing")
        self.context.session.commit()

    def analyse_form(self, form: Form):
        self.nb_form += 1
        phrase1_2 = ""
        if form.question_01_contrib1_answer is not None and form.question_01_contrib1_answer != "":
            phrase1_2 += form.question_01_contrib1_answer + "|"
        if form.question_01_contrib2_answer is not None and form.question_01_contrib2_answer != "":
            phrase1_2 += form.question_01_contrib2_answer + "|"
        if form.question_01_contrib3_answer is not None and form.question_01_contrib3_answer != "":
            phrase1_2 += form.question_01_contrib3_answer + "|"
        if form.question_02_contrib1_answer is not None and form.question_02_contrib1_answer != "":
            phrase1_2 += form.question_02_contrib1_answer + "|"
        if form.question_02_contrib2_answer is not None and form.question_02_contrib2_answer != "":
            phrase1_2 += form.question_02_contrib2_answer + "|"
        if form.question_02_contrib3_answer is not None and form.question_02_contrib3_answer != "":
            phrase1_2 += form.question_02_contrib3_answer + "|"
        phrase1_2 = phrase1_2.strip()
        phrase3_4 = ""
        if form.question_03_contrib1_answer is not None and form.question_03_contrib1_answer != "":
            phrase3_4 += form.question_03_contrib1_answer + "|"
        if form.question_03_contrib2_answer is not None and form.question_03_contrib2_answer != "":
            phrase3_4 += form.question_03_contrib2_answer + "|"
        if form.question_03_contrib3_answer is not None and form.question_03_contrib3_answer != "":
            phrase3_4 += form.question_03_contrib3_answer + "|"
        if form.question_04_contrib1_answer is not None and form.question_04_contrib1_answer != "":
            phrase3_4 += form.question_04_contrib1_answer + "|"
        if form.question_04_contrib2_answer is not None and form.question_04_contrib2_answer != "":
            phrase3_4 += form.question_04_contrib2_answer + "|"
        if form.question_04_contrib3_answer is not None and form.question_04_contrib3_answer != "":
            phrase3_4 += form.question_04_contrib3_answer + "|"
        phrase3_4 = phrase3_4.strip()
        self.make_stat(form, phrase1_2, phrase3_4)

    def make_stat(self, form: Form, phrase1_2: str, phrase3_4: str):
        form.stat = Stat()
        form.stat.q1_2_nb_word = self.model.nb_words(phrase1_2)
        if form.stat.q1_2_nb_word > 0:
            form.stat.q1_2_sentiment = self.model.sentiment(phrase1_2)
        form.stat.q3_4_nb_word = self.model.nb_words(phrase3_4)
        if form.stat.q3_4_nb_word > 0:
            form.stat.q3_4_sentiment = self.model.sentiment(phrase3_4)
        form.stat.pd_score = self.average(form.empathy_pd_6, form.empathy_pd_17, form.empathy_pd_24, form.empathy_pd_27)
        form.stat.pd_category = self.categorize(form.stat.pd_score)
        ec18 = None
        if form.empathy_ec_18 is not None and form.empathy_ec_18 != "*":
            ec18 = 6 - int(form.empathy_ec_18)
        form.stat.ec_score = self.average(form.empathy_ec_2, form.empathy_ec_9, ec18, form.empathy_ec_22)
        form.stat.ec_category = self.categorize(form.stat.ec_score)
        print(form.stat.q1_2_nb_word, phrase1_2[:50])








if __name__ == '__main__':
    art.tprint(config.name, "big")
    print("Textrank Service")
    print("================")
    print(f"V{config.version}")
    print(config.copyright)
    print()
    parser = argparse.ArgumentParser(description="Textrank Service")
    parser.add_argument("-e", "--echo", help="Sql Alchemy echo", action="store_true")
    args = parser.parse_args()
    context = Context()
    context.create(echo=args.echo)
    db_size = context.db_size()
    print(f"Database {context.db_name}: {db_size:.0f} Mb")
    a = TextrankService(context)
    a.analyse()
    print(f"Nb form: {a.nb_form}")
    new_db_size = context.db_size()
    print(f"Database {context.db_name}: {new_db_size:.0f} Mb")
    print(f"Database grows: {new_db_size - db_size:.0f} Mb ({((new_db_size - db_size) / db_size) * 100:.1f}%)")





