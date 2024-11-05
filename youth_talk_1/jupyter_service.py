import pandas as pd
import wordcloud


class JupyterService:

    def __init__(self, context):
        self.context = context
        self.df = None

    def get_by_sql(self, sql):
        if sql.strip().lower().startswith("select"):
            with self.context.engine.connect() as conn:
                self.df = pd.read_sql(sql, conn)
            return self.df
        return "Only for select statement"



