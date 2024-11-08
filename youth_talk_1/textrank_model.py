import datetime
import difflib
import unicodedata
from nltk.corpus import wordnet
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from dbcontext import Context
from sqlentities import Lema, Topic, Form, Stat, FormTopic
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
        self.r47 = 4 / 7
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
            if item != "" and "_" not in item:
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
        if "_" in word:
            return []
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

    def normalize_word(self, word: str):
        word = word.lower()
        word = self.strip_accents(word)
        word = word.replace("'", "")
        word = word.replace('"', "")
        word = word.replace("-", "")
        word = word.replace(".", "")
        word = word.replace(",", "")
        return word.strip()

    def split_phrases(self, text: str) -> list[str]:
        text = text.replace("|", ".")
        text = text.replace("\n", ".")
        text = text.replace("!", ".")
        text = text.replace(":", ".")
        text = text.replace("?", ".")
        text = text.replace(",", ".")
        text = text.replace("..", ".")
        return text.split(".")

    def split_words(self, s: str) -> list[str]:
        return s.split(" ")

    def split_lemas(self, s: str) -> list[Lema]:
        words = s.split(" ")
        res = []
        previous = None
        for word in words:
            word = self.normalize_word(word)
            if len(word) > 0:
                lema = Lema(word, previous)
                res.append(lema)
                previous = word
        return res

    def is_short(self, word: str) -> bool:
        if word in ["war", "gun", "bad"]:
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
        links = ["with", "which", "when", "under", "upper", "ever", "never", "less", "more", "true", "false", "other",
                 "have", "that", "thus", "then", "them", "being", "self", "either", "neither", "will", "still", "where",
                 "come", "they", "without", "very", "from", "after", "before", "this"]
        return s in links

    def remove_link_lemas(self, lemas: list[Lema]) -> list[Lema]:
        res = [lema for lema in lemas if not (self.is_link_word(lema.label))]
        return res

    def get_lema_equals(self, word, lemas: list[Lema]) -> Lema | None:
        for lema in lemas:
            if lema.label == word:
                return lema
        return None

    def get_sub_word47(self, word: str, lemas: list[Lema]) -> Lema | None:
        exceptions = ["wedding", "warming"]
        if word == "" or word is None:
            return None
        if word == "wedding":
            pass
        for lema in lemas:
            if "_" not in lema.label:
                w1 = word
                if w1 not in exceptions:
                    if w1.endswith("ing"):
                        w1 = w1[:-3]
                    elif w1.endswith("ed"):
                        w1 = w1[:2]
                w2 = lema.label
                if w2 not in exceptions:
                    if w2.endswith("ing"):
                        w2 = w2[:-3]
                    elif w2.endswith("ed"):
                        w2 = w2[:2]
                if ((w2.startswith(w1) and len(w2) / len(w1) > self.r47)
                        or (w1.startswith(w2) and len(w1) / len(w2) > self.r47)):
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

    def group_generic(self, lema: Lema, group_fn) -> Topic | None:
        for topic in self.topics.values():
            lema_res = group_fn(lema.label, topic.lemas)
            if lema_res is not None:
                topic.count += 1
                if group_fn == self.get_lema_equals:
                    lema_res.count += lema.count
                else:
                    topic.lemas.append(lema)
                return topic
        return None

    def group_lema_equals(self, lema: Lema) -> Topic | None:
        return self.group_generic(lema, self.get_lema_equals)

    def group_sub_word47(self, lema: Lema) -> Topic | None:
        return self.group_generic(lema, self.get_sub_word47)

    def group_synonyms(self, lema: Lema) -> Topic | None:
        return self.group_generic(lema, self.get_synonym)

    def group_gestalts(self, lema: Lema) -> Topic | None:
        return self.group_generic(lema, self.get_gestalt)

    def tokenize(self, text: str) -> list[list[Lema]]:
        text = self.normalize(text)
        phrases = self.split_phrases(text)
        lemass = [self.split_lemas(p) for p in phrases]  # Trier par count
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

    def sort_count_take(self, dico: dict[str, Lema], take=10) -> dict[str, Lema]:
        res: dict[str, Lema] = {}
        if len(dico) > 0:
            max_count = max([lema.count for lema in dico.values()])
            for i in range(max_count, 0, -1):
                lemas = [lema for lema in dico.values() if lema.count == i]
                for lema in lemas:
                    res[lema.label] = lema
                    take -= 1
                    if take == 0:
                        break
        return res


    def grouping(self, dico: dict[str, Lema], mode: str) -> list[Topic]:
        topics = []
        for key in dico.keys():
            if key in self.topics:
                topic = self.topics[key]
                topic.count += 1
                topic.lemas[0].count += dico[key].count
            else:
                topic = self.group_lema_equals(dico[key])
                if topic is None:
                    topic = self.group_sub_word47(dico[key])
                    if topic is None:
                        topic = self.group_gestalts(dico[key])
                        if topic is None:
                            topic = self.group_synonyms(dico[key])
                            if topic is None:
                                topic = Topic(label=key, source=mode, lemas=[dico[key]])
                                topic.count = 1
                                self.topics[key] = topic
            if topic not in topics:
                topics.append(topic)
            self.doubling(topic, dico[key])
        return topics

    # def doubling(self, topics: list[Topic]):
    #     for topic in topics:
    #         cloned = list(topic.lemas)
    #         for lema in cloned:
    #             if lema.label == "world":
    #                 pass
    #             if lema.previous is not None:
    #                 lema2 = Lema(f"{lema.previous}_{lema.label}")
    #                 if lema2 not in topic.lemas:
    #                     topic.lemas.append(lema2)
    #                 lema2 = [lema for lema in topic.lemas if lema == lema2][0]
    #                 lema2.count += 1
    #                 lema.previous = None

    def doubling(self, topic: Topic, lema: Lema):
        if lema.previous is not None:
            lema2 = Lema(f"{lema.previous}_{lema.label}")
            if lema2 not in topic.lemas:
                topic.lemas.append(lema2)
            lema2 = [lema for lema in topic.lemas if lema == lema2][0]
            lema2.count += 1
            lema2.previous = None


class TextrankService:

    def __init__(self, context):
        self.context = context
        self.model = TextrankModel()
        self.nb_form = 0
        self.nb_total_form = 0
        self.q1_2 = 12
        self.q3_4 = 34

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

    def make_stats(self):
        forms: list[Form] = self.context.session.execute(select(Form).where(Form.empathy_answers > 0)).scalars().all()
        for form in forms:
            if (form.question_01_contrib1_answer is not None
                    or form.question_01_contrib2_answer is not None
                    or form.question_01_contrib3_answer is not None):
                self.nb_form += 1
                phrase1_2 = self.get_phrase1_2(form)
                phrase3_4 = self.get_phrase3_4(form)
                self.make_stat(form, phrase1_2, phrase3_4)
        print("Committing")
        self.context.session.commit()

    def get_phrase1_2(self, form: Form, sep=". ") -> str:
        phrase1_2 = ""
        if form.question_01_contrib1_answer is not None and form.question_01_contrib1_answer != "":
            phrase1_2 += form.question_01_contrib1_answer + sep
        if form.question_01_contrib2_answer is not None and form.question_01_contrib2_answer != "":
            phrase1_2 += form.question_01_contrib2_answer + sep
        if form.question_01_contrib3_answer is not None and form.question_01_contrib3_answer != "":
            phrase1_2 += form.question_01_contrib3_answer + sep
        if form.question_02_contrib1_answer is not None and form.question_02_contrib1_answer != "":
            phrase1_2 += form.question_02_contrib1_answer + sep
        if form.question_02_contrib2_answer is not None and form.question_02_contrib2_answer != "":
            phrase1_2 += form.question_02_contrib2_answer + sep
        if form.question_02_contrib3_answer is not None and form.question_02_contrib3_answer != "":
            phrase1_2 += form.question_02_contrib3_answer + sep
        return phrase1_2.strip()

    def get_phrase3_4(self, form: Form, sep=". ") -> str:
        phrase3_4 = ""
        if form.question_03_contrib1_answer is not None and form.question_03_contrib1_answer != "":
            phrase3_4 += form.question_03_contrib1_answer + sep
        if form.question_03_contrib2_answer is not None and form.question_03_contrib2_answer != "":
            phrase3_4 += form.question_03_contrib2_answer + sep
        if form.question_03_contrib3_answer is not None and form.question_03_contrib3_answer != "":
            phrase3_4 += form.question_03_contrib3_answer + sep
        if form.question_04_contrib1_answer is not None and form.question_04_contrib1_answer != "":
            phrase3_4 += form.question_04_contrib1_answer + sep
        if form.question_04_contrib2_answer is not None and form.question_04_contrib2_answer != "":
            phrase3_4 += form.question_04_contrib2_answer + sep
        if form.question_04_contrib3_answer is not None and form.question_04_contrib3_answer != "":
            phrase3_4 += form.question_04_contrib3_answer + sep
        return phrase3_4.strip()

    def make_stat(self, form: Form, phrase1_2: str, phrase3_4: str):
        form.stat = Stat()
        form.stat.date = datetime.datetime.now()
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
        form.stat.pt_score = self.average(form.empathy_pt_8, form.empathy_pt_11, form.empathy_pt_25, form.empathy_pt_28)
        form.stat.pt_category = self.categorize(form.stat.pt_score)
        form.stat.f_score = self.average(form.empathy_f_5, form.empathy_f_16, form.empathy_f_23, form.empathy_f_26)
        form.stat.f_category = self.categorize(form.stat.f_score)
        form.stat.empathy_score = self.average(form.stat.pd_score, form.stat.ec_score,
                                               form.stat.pt_score, form.stat.f_score)
        form.stat.empathy_category = self.categorize(form.stat.empathy_score)
        print(form.stat.q1_2_nb_word, phrase1_2[:50], form.stat.q3_4_nb_word, phrase3_4[:50])

    def make_questions_textrank(self, mode: str, question=12):  # question=12 or 34
        print(f"Textranking Q1 Q2 in mode {mode}")
        self.load_topics(mode)
        date_condition = Stat.textrank_date.is_(None)
        if mode == "nltk":
            date_condition = Stat.nltk_date.is_(None)
        if question == 12:
            q_condition = Stat.q1_2_nb_word > 0
        else:
            q_condition = Stat.q3_4_nb_word > 0
        forms: list[Form] = self.context.session.execute(
            select(Form).join(Stat).options(joinedload(Form.stat))
            .where((Form.empathy_answers > 0) & date_condition & q_condition)
        ).scalars().all()
        self.nb_total_form = len(forms)
        self.nb_form = 0
        print(f"Textranking {self.nb_total_form} forms")
        for form in forms[:]:
            if question == 12:
                phrase = self.get_phrase1_2(form)
            else:
                phrase = self.get_phrase3_4(form)
            if mode == "textrank":
                lemass = self.model.tokenize_textrank(phrase)
            else:
                lemass = self.model.tokenize(phrase)
            dico = self.model.count(lemass)
            if mode == "nltk":
                dico = self.model.sort_count_take(dico)
            topics = self.model.grouping(dico, mode)
            print(self.nb_form, phrase[:100], topics)
            for topic in topics:
                form_topic = FormTopic(topic, question)
                if form.form_topics is None:
                    form.form_topics = [form_topic]
                else:
                    form.form_topics.append(form_topic)
            self.nb_form += 1
            if mode == "textrank":
                form.stat.textrank_date = datetime.datetime.now()
            elif mode == "nltk":
                form.stat.nltk_date = datetime.datetime.now()
        self.context.session.commit()

    def load_topics(self, mode: str):
        print("Loading topics")
        topics = self.context.session.execute(
            select(Topic).options(joinedload(Topic.lemas)).where(Topic.source == mode)).unique().scalars().all()
        for topic in topics:
            self.model.topics[topic.label] = topic
        print(f"{len(self.model.topics)} topics in cache")

    def rename_topics(self):
        topics: list[Topic] = self.context.session.execute(
            select(Topic).options(joinedload(Topic.lemas))).unique().scalars().all()
        nb = 0
        for topic in topics:
            if len(topic.lemas) > 0:
                max_count = max([lema.count for lema in topic.lemas])
                lema = [lema for lema in topic.lemas if lema.count == max_count][0]
                if len(str(lema.label)) < 15:
                    topic.label = lema.label
                nb += 1
        print(f"Rename {nb} topics")
        self.context.session.commit()


if __name__ == '__main__':
    art.tprint(config.name, "big")
    print("Textrank Service")
    print("================")
    print(f"V{config.version}")
    print(config.copyright)
    print()
    context = Context()
    context.create(echo=False)
    db_size = context.db_size()
    print(f"Database {context.db_name}: {db_size:.0f} Mb")
    a = TextrankService(context)
    # a.make_stats()
    # a.make_questions_textrank("textrank", 12)
    # a.make_questions_textrank("nltk", 12)
    a.rename_topics()
    print(f"Nb form: {a.nb_form}")
    new_db_size = context.db_size()
    print(f"Database {context.db_name}: {new_db_size:.0f} Mb")
    print(f"Database grows: {new_db_size - db_size:.0f} Mb ({((new_db_size - db_size) / db_size) * 100:.1f}%)")





