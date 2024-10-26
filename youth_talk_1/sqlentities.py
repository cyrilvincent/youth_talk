from sqlalchemy import Column, ForeignKey, Boolean, UniqueConstraint, Table, Index, DateTime
from sqlalchemy.types import BigInteger, Integer, String, Float, Date, SmallInteger
from sqlalchemy.orm import relationship, Mapped, mapped_column
from dbcontext import Base

# form 1-* form_topic  *-1 topic -* lema
#       -1 stat


class Form(Base):
    __tablename__ = "form"

    id = Column(Integer, primary_key=True)
    question_01_contrib1_answer = Column(String())
    question_02_contrib1_answer = Column(String())
    question_03_contrib1_answer = Column(String())
    form_topics: Mapped[list["FormTopic"]] = relationship(back_populates="form")
    date_added = Column(DateTime, nullable=False)
    date_computed = Column(DateTime)

    def __repr__(self):
        return f"{self.id}"


class Topic(Base):
    __tablename__ = "topic"

    id = Column(BigInteger, primary_key=True)
    label = Column(String(50), nullable=False, index=True)
    form_topics: Mapped[list["FormTopic"]] = relationship(back_populates="topic")
    stat: Mapped["Stat"] = relationship(back_populates="topic", uselist=False)
    lemas: Mapped[list["Lema"]] = relationship(back_populates="topic")
    count = Column(Integer, nullable=False)
    source = Column(String(5), nullable=False, index=True)
    date = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"{self.label} {self.count} {self.lemas}"


class FormTopic(Base):
    __tablename__ = "form_topic"

    id = Column(Integer, primary_key=True)
    form_id = Column(ForeignKey('form.id'), nullable=False)
    form: Mapped[Form] = relationship(back_populates="form_topics")
    topic_id = Column(ForeignKey('topic.id'), nullable=False)
    topic: Mapped[Topic] = relationship(back_populates="form_topics")
    count = Column(Integer, nullable=False)
    date = Column(DateTime, nullable=False)

    __table_args__ = (UniqueConstraint('form_id', 'topic_id'),)

    @property
    def key(self):
        return self.form_id, self.topic_id

    def __repr__(self):
        return f"{self.id} {self.form_id} {self.topic_id}"


class Lema(Base):
    __tablename__ = "lema"

    id = Column(BigInteger, primary_key=True)
    label = Column(String(), nullable=False)
    topic_id = Column(ForeignKey('topic.id'), nullable=False)
    topic: Mapped[Topic] = relationship(back_populates="lemas")
    previous = Column(String())  # PAs grave si mappé

    def __repr__(self):
        return f"{self.label}"

    def __eq__(self, other):
        return self.label == other.label


class Stat(Base):
    __tablename__ = "stat"

    id = Column(ForeignKey('topic.id'), primary_key=True)
    pd_score = Column(Float)
    pd_category = Column(Integer)
    # topic_id = Column(ForeignKey('topic.id'), nullable=False)
    topic: Mapped[Topic] = relationship(back_populates="stat")

    def __repr__(self):
        return f"{self.id} {self.pd_category}"


