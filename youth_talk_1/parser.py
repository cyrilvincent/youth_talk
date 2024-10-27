import datetime
import art
import pandas as pd
from sqlalchemy import select, text
import config
import argparse
from dbcontext import Context


class YT1Parser:

    def __init__(self, context: Context):
        self.context = context
        self.df = None

    def parse(self, file: str):
        path = f"{config.path}/{file}"
        print(f"Load {path}")
        self.df = pd.read_csv(path, index_col="participant_id", low_memory=False)
        self.df.index.names = ['id']
        self.df.columns = self.df.columns.str.lower()
        self.df["date_added"] = [datetime.datetime.now()] * len(self.df)
        self.df["date_computed"] = [None] * len(self.df)
        print(self.df)
        self.context.create_engine()
        with context.engine.begin() as connection:
            print("Saving to SQL")
            self.df.to_sql("form", connection, if_exists="replace", index=True)
            connection.execute(text("ALTER TABLE form ADD PRIMARY KEY (id)"))

    # def get_all(self) -> list[Form]:
    #     l = self.context.session.execute(select(Form)).scalars().first()
    #     for e in l:
    #         print(e.id)



if __name__ == '__main__':
    art.tprint(config.name, "big")
    print("Youth Talk 1 Parser")
    print("===================")
    print(f"V{config.version}")
    print(config.copyright)
    print()
    parser = argparse.ArgumentParser(description="Gdlet Parser")
    parser.add_argument("-e", "--echo", help="Sql Alchemy echo", action="store_true")
    args = parser.parse_args()
    context = Context()
    context.create(echo=args.echo)
    db_size = context.db_size()
    print(f"Database {context.db_name}: {db_size:.0f} Mb")
    p = YT1Parser(context)
    p.parse("YT_dataset-v1.csv")
    # p.get_all()
    new_db_size = context.db_size()
    print(f"Database {context.db_name}: {new_db_size:.0f} Mb")
    print(f"Database grows: {new_db_size - db_size:.0f} Mb ({((new_db_size - db_size) / db_size) * 100:.1f}%)")


