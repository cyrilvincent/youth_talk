from sqlalchemy import Column, ForeignKey, Boolean, UniqueConstraint, Table, Index, DateTime
from sqlalchemy.types import BigInteger, Integer, String, Float, Date, SmallInteger
from sqlalchemy.orm import relationship, Mapped, mapped_column
from dbcontext import Base

# form 1-* from_topic  *-1 top -* word


class FormInstance(Base):
    __tablename__ = "form_instance"

    id = Column(Integer, primary_key=True)
    q1 = Column(String(), nullable=False)
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
    date = Column(DateTime, nullable=False)

    def __repr__(self):
        return f"{self.id} {self.label}"


class FormTopic(Base):
    __tablename__ = "form_topic"

    id = Column(Integer, primary_key=True)
    form_id = Column(ForeignKey('form.id'), nullable=False)
    form: Mapped[FormInstance] = relationship(back_populates="form_topics")
    topic_id = Column(ForeignKey('topic.id'), nullable=False)
    topic: Mapped[Topic] = relationship(back_populates="form_topics")
    count = Column(Integer, nullable=False)
    words: Mapped[list["Word"]] = relationship(back_populates="topic")
    date = Column(DateTime, nullable=False)

    __table_args__ = (UniqueConstraint('form_id', 'topic_id'),)

    @property
    def key(self):
        return self.form_id, self.topic_id

    def __repr__(self):
        return f"{self.id} {self.form_id} {self.topic_id}"


class Word(Base):
    __tablename__ = "word"

    id = Column(BigInteger, primary_key=True)
    label = Column(String(), nullable=False)
    topic_id = Column(ForeignKey('topic.id'), nullable=False)
    topic: Mapped[Topic] = relationship(back_populates="words")

    def __repr__(self):
        return f"{self.id} {self.label}"

