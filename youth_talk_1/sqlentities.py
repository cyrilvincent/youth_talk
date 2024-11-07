import datetime

from sqlalchemy import Column, ForeignKey, Boolean, UniqueConstraint, Table, Index, DateTime
from sqlalchemy.types import BigInteger, Integer, String, Float, Date, SmallInteger
from sqlalchemy.orm import relationship, Mapped, mapped_column
from dbcontext import Base

# form 1-* form_topic  *-1 topic -* lema
#       -1 stat
#       -* llm


class Form(Base):
    __tablename__ = "form"

    id = Column(Integer, primary_key=True)
    question_01_contrib1_answer = Column(String())
    question_01_contrib2_answer = Column(String())
    question_01_contrib3_answer = Column(String())
    question_02_contrib1_answer = Column(String())
    question_02_contrib2_answer = Column(String())
    question_02_contrib3_answer = Column(String())
    question_03_contrib1_answer = Column(String())
    question_03_contrib2_answer = Column(String())
    question_03_contrib3_answer = Column(String())
    question_04_contrib1_answer = Column(String())
    question_04_contrib2_answer = Column(String())
    question_04_contrib3_answer = Column(String())
    empathy_answers = Column(BigInteger(), nullable=False)
    empathy_pd_6 = Column(Integer)
    empathy_pd_17 = Column(Integer)
    empathy_pd_24 = Column(Integer)
    empathy_pd_27 = Column(Integer)
    empathy_ec_2 = Column(Integer)
    empathy_ec_9 = Column(Integer)
    empathy_ec_18 = Column(String(1))
    empathy_ec_22 = Column(Integer)
    empathy_pt_8 = Column(Integer)
    empathy_pt_11 = Column(Integer)
    empathy_pt_25 = Column(Integer)
    empathy_pt_28 = Column(Integer)
    empathy_f_5 = Column(Integer)
    empathy_f_16 = Column(Integer)
    empathy_f_23 = Column(Integer)
    empathy_f_26 = Column(Integer)
    form_topics: Mapped[list["FormTopic"]] = relationship(back_populates="form")
    date_added = Column(DateTime, nullable=False)
    date_computed = Column(DateTime)
    stat: Mapped["Stat"] = relationship(back_populates="form", uselist=False)
    llms: Mapped[list["LLM"]] = relationship(back_populates="form")

    def __repr__(self):
        return f"{self.id}"


class Topic(Base):
    __tablename__ = "topic"

    id = Column(BigInteger, primary_key=True)
    label = Column(String(50), nullable=False, index=True)
    form_topics: Mapped[list["FormTopic"]] = relationship(back_populates="topic")
    lemas: Mapped[list["Lema"]] = relationship(back_populates="topic")
    count = Column(Integer, nullable=False)
    source = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False)

    def __init__(self, label=None, source=None, lemas=[]):
        self.label = label[:50]
        self.source = source
        self.lemas = lemas
        self.date = datetime.datetime.now()
        self.count = 0

    def __repr__(self):
        return f"{self.label} {self.count} {self.lemas}"


class FormTopic(Base):
    __tablename__ = "form_topic"

    id = Column(Integer, primary_key=True)
    form_id = Column(ForeignKey('form.id'), nullable=False)
    form: Mapped[Form] = relationship(back_populates="form_topics")
    topic_id = Column(ForeignKey('topic.id'), nullable=False)
    topic: Mapped[Topic] = relationship(back_populates="form_topics")
    question_nb = Column(Integer, nullable=False)  #12 or 34
    score = Column(Float)
    date = Column(DateTime, nullable=False)

    __table_args__ = (UniqueConstraint('form_id', 'topic_id', 'question_nb'),)

    def __init__(self, topic: Topic, question_nb: int):
        self.topic = topic
        self.question_nb = question_nb
        self.date = datetime.datetime.now()

    @property
    def key(self):
        return self.form_id, self.topic_id

    def __repr__(self):
        return f"{self.id} {self.form_id} {self.topic_id}"


class Lema(Base):
    __tablename__ = "lema"

    id = Column(BigInteger, primary_key=True)
    label = Column(String(), nullable=False)
    count = Column(Integer, nullable=False)
    topic_id = Column(ForeignKey('topic.id'), nullable=False)
    topic: Mapped[Topic] = relationship(back_populates="lemas")
    date = Column(DateTime, nullable=False)

    def __init__(self, label=None, previous=None):
        self.label = label
        self.previous: str | None = previous
        self.count = 0
        self.date = datetime.datetime.now()

    def __repr__(self):
        return f"{self.label} {self.count}"

    def __eq__(self, other):
        return self.label == other.label

    def __hash__(self):
        return self.label.__hash__()


class Stat(Base):
    __tablename__ = "stat"

    id = Column(ForeignKey('form.id'), primary_key=True)
    pd_score = Column(Float)
    pd_category = Column(SmallInteger)
    ec_score = Column(Float)
    ec_category = Column(SmallInteger)
    pt_score = Column(Float)
    pt_category = Column(SmallInteger)
    f_score = Column(Float)
    f_category = Column(SmallInteger)
    empathy_score = Column(Float)
    empathy_category = Column(SmallInteger)
    q1_2_nb_word = Column(Integer)
    q1_2_sentiment = Column(Float)
    q3_4_nb_word = Column(Integer)
    q3_4_sentiment = Column(Float)
    date = Column(DateTime, nullable=False)
    textrank_date = Column(DateTime) # A dupliquer pour les q34
    nltk_date = Column(DateTime)
    td_idf_date = Column(DateTime)
    openai_date = Column(DateTime)
    form: Mapped[Form] = relationship(back_populates="stat")

    def __repr__(self):
        return f"{self.id} {self.q1_2_nb_word} {self.q1_2_sentiment}"


class LLM(Base):
    __tablename__ = "llm"

    id = Column(BigInteger, primary_key=True)
    text = Column(String(255), nullable=False)
    source = Column(String(20), nullable=False)
    form_id = Column(ForeignKey('form.id'), nullable=False)
    form: Mapped[Form] = relationship(back_populates="llms")
    question_nb = Column(Integer, nullable=False)
    date = Column(DateTime, nullable=False)

    __table_args__ = (UniqueConstraint('form_id', 'source', 'question_nb'),)

    def __init__(self, text=None, source=None, question=None):
        self.text = text
        self.source = source
        self.question_nb = question
        self.date = datetime.datetime.now()

    def __repr__(self):
        return f"{self.text}"


