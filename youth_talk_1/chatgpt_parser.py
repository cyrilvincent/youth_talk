from sqlalchemy import text

from dbcontext import Context
import pandas as pd

def normalize(word):
    word = word.replace("_", " ").lower()
    return word

context = Context()
context.create(echo=False)
# df = pd.read_json("data/chatgpt.json")
# df.insert(0, 'id', range(len(df)))
# df["question_nb"] = [12] * len(df)
# df.to_csv("data/chatgpt.csv")
df = pd.read_csv("data/chatgpt.csv")
df["index"]=df["index"].apply(normalize)
context.create_engine()
with context.engine.begin() as connection:
    df.to_sql("chatgpt_highlevel", connection, if_exists="replace", index=True)
    connection.execute(text("ALTER TABLE chatgpt_highlevel ADD PRIMARY KEY (id)"))

