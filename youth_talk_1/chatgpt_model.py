import datetime

import os
from openai import OpenAI # pip install openai
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from dbcontext import Context
from jupyter_service import JupyterService
from sqlentities import Stat, Form, LLM, Topic, FormTopic, Lema, GPTComment, GPTHighLevel
from textrank_model import TextrankService


class ChatGPTModel:

    def __init__(self, chat_model="gpt-4o-mini"):
        print(f"OpenAI {chat_model}")
        with open("data/openai.env") as f:
            key = f.read()
        self.client = OpenAI(api_key=key)
        self.chat_model = chat_model
        self.dico: dict[str, str] = {
            "ec": "empathic concern",
            "pt": "perspective taking",
            "f": "fantasy",
            "pd": "personal distress",
        }
        self.limit = 128000

    def topic(self, text: str, nb=3) -> str:
        topic = f"What are the {nb} main topics, one simple word each, of the following message, in CSV format, in english?"
        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": topic},
                      {"role": "user", "content": text}])
        return completion.choices[0].message.content

    def empathy(self, text: str, topic: str, category: str, positive: bool):
        text = text[:self.limit]
        positive_text = "positive" if positive else "negative"
        question = f"Explain to me the {positive_text} terms related to {self.dico[category]} and {topic} in less than 20 words"
        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": question},
                      {"role": "user", "content": text}])
        res = completion.choices[0].message.content
        text = text.replace("\n", ". ")[:50]
        print(f"{question} {text} => {res}")
        return res

    def high_level_topics(self, text: str, category: str, positive: bool):
        text = text[:self.limit]
        positive_text = "positive" if positive else "negative"
        question = f"What are the {positive_text} topics related to {self.dico[category]} in the following text? Ponderate each topic by a score from 0 to 100 in json format"
        completion = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "system", "content": question},
                      {"role": "user", "content": text}])
        res = completion.choices[0].message.content
        print(f"{question} length: {len(text)} => {res}")
        return res


class LLMService(TextrankService):

    def __init__(self, context):
        super().__init__(context)
        self.gpt_model = ChatGPTModel()
        self.topics: dict[str, Topic] = {}
        self.lemas: dict[str, Lema] = {}
        self.gpt_comments: dict[str, GPTComment] = {}

    def gpt(self, question):
        if question == 12:
            q_condition = Stat.q1_2_nb_word > 0
        else:
            q_condition = Stat.q3_4_nb_word > 0
        date_condition = Stat.openai_date.is_(None)
        forms: list[Form] = self.context.session.execute(
            select(Form).join(Stat).options(joinedload(Form.stat))
            .where((Form.empathy_answers > 0) & date_condition & q_condition)
        ).scalars().all()
        self.nb_total_form = len(forms)
        for form in forms:
            if question == 12:
                phrase = self.get_phrase1_2(form)
            else:
                phrase = self.get_phrase3_4(form)
            llm = self.gpt_form(question, phrase)
            if llm is not None:
                print(f"{self.nb_form}/{self.nb_total_form} {llm.text} ({llm.source}) <= {phrase:255}")
                if form.llms is None:
                    form.llms = [llm]
                else:
                    form.llms.append(llm)
            form.stat.openai_date = datetime.datetime.now()
            self.context.session.commit()
            self.nb_form += 1

    def gpt_form(self, question: int, phrase: str) -> LLM | None:
        lemass = self.model.tokenize(phrase)
        dico = self.model.count(lemass)
        length = len(dico)
        if length == 0:
            return None
        if length == 1:
            return LLM(list(dico.values())[0].label, "nltk", question)
        elif length < 6:
            dico = self.model.sort_count_take(dico, min(3, length))
            words = [lema.label for lema in dico.values()]
            csv = ",".join(words)
            llm = LLM(csv, "nltk", question)
            return llm
        return self.chat_form(question, phrase)

    def chat_form(self, question, text: str) -> LLM:
        res = self.gpt_model.topic(text).strip()
        return LLM(res, self.gpt_model.chat_model, question)

    def create_topics(self, question: int):
        print("Querying topics")
        topics: list[Topic] = self.context.session.execute(
            select(Topic).where(Topic.source == self.gpt_model.chat_model)
        ).scalars().all()
        for topic in topics:
            self.topics[topic.label] = topic
        print(f"Found {len(self.topics)} topics")
        llms: list[LLM] = self.context.session.execute(
            select(LLM).options(joinedload(LLM.form)).where(LLM.question_nb == question)
        ).scalars().all()
        self.nb_total_form = len(llms)
        self.nb_form = 0
        for llm in llms:
            self.create_form_topics(question, llm)
            self.nb_form += 1
        print("Comitting")
        self.context.session.commit()

    def create_form_topics(self, question, llm: LLM):
        words = llm.text.split(",")
        words = [word.lower().strip() for word in words]
        for word in words:
            if len(word) > 0:
                if word not in self.topics:
                    topic = Topic(word, self.gpt_model.chat_model)
                    self.topics[word] = topic
                    print(f"{self.nb_form} => {word}")
                topic = self.topics[word]
                topic.count += 1
                form_topic = FormTopic(topic, question)
                if llm.form.form_topics is None:
                    llm.form.form_topics = [form_topic]
                else:
                    llm.form.form_topics.append(form_topic)

    def create_synonyms(self):
        print(f"Quering Lemas")
        lemas: list[Lema] = self.context.session.execute(select(Lema)).scalars().all()
        for lema in lemas:
            if lema.label not in self.lemas:
                self.lemas[lema.label] = lema
        print(f"Found {len(self.lemas)} unique lemas")
        print("Querying Topics")
        topics: list[Topic] = self.context.session.execute(
            select(Topic).options(joinedload(Topic.lemas)).where(Topic.source == self.gpt_model.chat_model)
        ).unique().scalars().all()
        print(f"Found {len(topics)} topics")
        nb_topic = 0
        nb_total_topic = len(topics)
        nb_lema = 0
        for topic in topics:
            nb_topic += 1
            synonyms = self.model.get_synonyms(topic.label)
            if topic.label not in synonyms:
                synonyms.insert(0, topic.label)
            for synonym in synonyms[:10]:
                if synonym in self.lemas:
                    lema = Lema(synonym)
                    lema.count = self.lemas[synonym].count
                    if lema not in topic.lemas:
                        topic.lemas.append(lema)
                        nb_lema += 1
                        print(f"{nb_topic}/{nb_total_topic} => Lema {nb_lema} {synonym}")
        print("Comitting")
        self.context.session.commit()

    def get_topic(self, topic_id: int) -> Topic:
        topic = self.context.session.execute(
            select(Topic).options(joinedload(Topic.lemas)).options(joinedload(Topic.form_topics))
            .where(Topic.id == topic_id)).unique().scalars().first()
        return topic

    def get_forms_by_topic_id(self, topic_id):
        topic = self.context.session.execute(
            select(Topic).options(joinedload(Topic.form_topics).joinedload(FormTopic.form))
            .where(Topic.id == topic_id)).unique().scalars().first()
        forms = [form_topic.form for form_topic in topic.form_topics]
        return forms

    def merge_topics(self, from_id: int, to_id: int):
        topic_from = self.get_topic(from_id)
        topic_to = self.get_topic(to_id)
        print(f"Merging {topic_from.label} => {topic_to.label}")
        for lema in list(topic_from.lemas):
            if lema not in topic_to.lemas:
                topic_to.lemas.append(lema)
        topic_from.lemas.clear()
        for form_topic in list(topic_from.form_topics):
            res = [(form_topic.form_id, form_topic.question_nb) for form_topic in topic_to.form_topics]
            if (form_topic.form_id, form_topic.question_nb) not in res:
                topic_to.form_topics.append(form_topic)
                form_topic.topic = topic_to
            else:
                self.context.session.delete(form_topic)
        topic_from.form_topics.clear()
        topic_to.count += topic_from.count
        print("Comitting")
        self.context.session.delete(topic_from)
        self.context.session.commit()

    def topic_empathies(self, question: int):
        gpt_comments = self.context.session.execute(select(GPTComment)).scalars().all()
        for gpt_comment in gpt_comments:
            self.gpt_comments[gpt_comment.key] = gpt_comment
        jupyter = JupyterService(self.context)
        for key in self.gpt_model.dico.keys():
            for positive in [True, False]:
                df = jupyter.get_scores(self.gpt_model.chat_model, question, key, positive, 5, 0)
                for index, row in df.iterrows():
                    topic_id = row["id"]
                    if (question, key, positive, topic_id) not in self.gpt_comments:
                        topic = self.get_topic(topic_id)
                        self.topic_empathies_categories(question, key, positive, topic)

    def topic_highlevel_empathies(self, question: int):
        gpt_comments = self.context.session.execute(select(GPTComment)).scalars().all()
        for gpt_comment in gpt_comments:
            self.gpt_comments[gpt_comment.key] = gpt_comment
        for key in self.gpt_model.dico.keys():
            for positive in [True, False]:
                topics: list[Topic] = self.context.session.execute(
                    select(Topic).where(Topic.source == "highlevel")).scalars().all()
                for topic in topics:
                    if (question, key, positive, topic.id) not in self.gpt_comments:
                        topic = self.get_topic(topic.id)
                        self.topic_empathies_categories(question, key, positive, topic)

    def topic_empathies_categories(self, question: int, empathy: str, positive: bool, topic: Topic):
        forms = self.get_forms_by_topic_id(topic.id)
        text = ""
        for form in forms:
            if question == 12:
                phrase = self.get_phrase1_2(form)
            else:
                phrase = self.get_phrase3_4(form)
            text += phrase + "\n"
        res = self.gpt_model.empathy(text, topic.label, empathy, positive)
        gpt_comment = GPTComment(topic, question, empathy, positive, res, self.gpt_model.chat_model)
        self.context.session.add(gpt_comment)
        self.context.session.commit()

    def create_highlevel_topics(self, question: int):
        dico: dict[tuple[str, bool], list[Form]] = {}
        for key in self.gpt_model.dico.keys():
            for positive in [True, False]:
                dico[key, positive] = []
        forms: list[Form] = self.context.session.execute(
            select(Form).options(joinedload(Form.stat)).join(Stat).where(Form.stat != None)).scalars().all()
        for form in forms:
            if form.stat.pd_category == 0:
                dico["pd", False].append(form)
            elif form.stat.pd_category == 2:
                dico["pd", True].append(form)
            elif form.stat.ec_category == 0:
                dico["ec", False].append(form)
            elif form.stat.ec_category == 2:
                dico["ec", True].append(form)
            elif form.stat.pt_category == 0:
                dico["pt", False].append(form)
            elif form.stat.pt_category == 2:
                dico["pt", True].append(form)
            elif form.stat.f_category == 0:
                dico["f", False].append(form)
            elif form.stat.f_category == 2:
                dico["f", True].append(form)
        for key in self.gpt_model.dico.keys():
            for positive in [True, False]:
                forms = dico[key, positive]
                text = ""
                for form in forms:
                    if question == 12:
                        phrase = self.get_phrase1_2(form)
                    else:
                        phrase = self.get_phrase3_4(form)
                    text += phrase + "\n"
                print(key, positive)
                self.gpt_model.high_level_topics(text, key, positive)

    def create_highlevel_linked_topic(self, question: int):
        dico: dict[str, GPTHighLevel] = {}
        highlevels: list[GPTHighLevel] = self.context.session.execute(
            select(GPTHighLevel).where(GPTHighLevel.question_nb == question)).scalars().all()
        for highlevel in highlevels:
            if highlevel.index not in dico:
                dico[highlevel.index] = highlevel
                topic = Topic(highlevel.index, "highlevel")
                self.context.session.add(topic)
        self.context.session.commit()








if __name__ == '__main__':
    print("OpenAI test")
    # Pricing for 1M input / ouput
    # 4o-mini 0.15 / 0.6
    # 4o 2.5 / 10
    # 3.5-turbo 1 / 2

    time0 = datetime.datetime.now()
    # openai = ChatGPTModel("gpt-4o-mini")
    # shakespear = "Let me not to the marriage of true minds. Admit impediments. Love is not love. Which alters when it alteration finds. Or bends with the remover to remove. O no! it is an ever-fixed mark"
    # tanzania = "am in Tanzania, but I am a Congomani scout of fesco, and association de scouts du sud kivu, district scout du lac Tanganyika, for my part, I wanted to talk about the south of Congomani scouts in Tanzania who fled, I mean refugees, from the Congo who are in Tanzania, on the Tanzanian side I grew up I request the World Scout Association to try to contact the TSA Tanzania Scout Association for permission to be given to Congomani scouts present in the Nyarugusu camp to continue scout movements and to grow. The scouts grew up from the beginning of the camp but in recent years they have been tamed without any reason and they have tried asking for permission reached other young people to travel to the scout headquarters in Tanzania TSA to ask for help with the matter now they have received answers every time they try to remind you they give a promise we will work on it today the camp leaders wrote a letter saying that in order for a scout to be in the camp a letter is required from for the minister of interior to give permission to the existing group in the camp now what is going on in the nyarugusu camp, there are more than 900000 young people and children and there are more than 545 scouts, the young people are suffering from not having a group they like."    # text = openai.topic(shakespear)
    # rich = "I want to be a billionaire"
    # text = openai.topic(rich)
    # print(text)
    context = Context()
    context.create(echo=False)
    db_size = context.db_size()
    print(f"Database {context.db_name}: {db_size:.0f} Mb")
    s = LLMService(context)
    # s.gpt(12)
    # s.create_topics(12)
    # s.create_synonyms()
    # s.topic_empathies(12)
    # s.create_highlevel_topics(12)
    # s.create_highlevel_linked_topic(12)
    s.topic_highlevel_empathies(12)
    # s.merge_topics(30897, 30795)
    print(f"Nb form: {s.nb_form}")
    print(f"Nb topics: {len(s.topics)}")
    new_db_size = context.db_size()
    print(f"Database {context.db_name}: {new_db_size:.0f} Mb")
    print(f"Total time {datetime.datetime.now() - time0}")












