import datetime

import art
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import select
from sqlalchemy.orm import joinedload

import config
from dbcontext import Context
from sqlentities import Stat, Form, Topic, Lema, FormTopic
from textrank_model import TextrankService
import pandas as pd

class TdIdfService(TextrankService):

    def __init__(self, context):
        super().__init__(context)
        self.vectorizer = TfidfVectorizer()

    def get_phrase(self, form: Form, question: int) -> str:
        if question == 12:
            return self.get_phrase1_2(form)
        return self.get_phrase3_4(form)

    def make_questions_td_idf(self, question=12):
        print(f"TdIdf Q1 Q2")
        self.load_topics("tdidf")
        date_condition = Stat.td_idf_date.is_(None)
        if question == 12:
            q_condition = Stat.q1_2_nb_word > 0
        else:
            q_condition = Stat.q3_4_nb_word > 0
        print("Querying")
        forms: list[Form] = self.context.session.execute(
            select(Form).join(Stat).options(joinedload(Form.stat)).options(joinedload(Form.form_topics))
            .where((Form.empathy_answers > 0) & date_condition & q_condition)
        ).unique().scalars().all()
        documents = [self.get_phrase(form, question) for form in forms]
        print("TdIdfing")
        vector = self.vectorizer.fit_transform(documents)
        columns = self.vectorizer.get_feature_names_out()
        print("Pandasing")
        df = pd.DataFrame(vector.toarray(), columns=columns)
        i = 0
        for _, row in list(df.iterrows())[:]:
            for col in columns:
                if row[col] > 0.1 and not self.model.is_short(col) and not self.model.is_link_word(col):
                    print(i, col, row[col])
                    if col in self.model.topics:
                        topic = self.model.topics[col]
                    else:
                        topic = Topic(col, "tdidf", [Lema(col)])
                    topic.count += 1
                    form_topic = FormTopic(topic, question)
                    form_topic.score = float(row[col])
                    forms[i].form_topics.append(form_topic)
                    forms[i].stat.td_idf_date = datetime.datetime.now()
            i += 1
        self.context.session.commit()




if __name__ == '__main__':
    art.tprint(config.name, "big")
    print("TdIdf Service")
    print("=============")
    print(f"V{config.version}")
    print(config.copyright)
    print()
    context = Context()
    context.create(echo=False)
    db_size = context.db_size()
    print(f"Database {context.db_name}: {db_size:.0f} Mb")
    s = TdIdfService(context)
    s.make_questions_td_idf(12)
    new_db_size = context.db_size()
    print(f"Database {context.db_name}: {new_db_size:.0f} Mb")
    print(f"Database grows: {new_db_size - db_size:.0f} Mb ({((new_db_size - db_size) / db_size) * 100:.1f}%)")

