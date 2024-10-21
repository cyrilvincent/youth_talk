from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
import config

Base = declarative_base()


class Context:

    def __init__(self, connection_string=config.connection_string):
        self.engine: Engine | None = None
        self.session: Session | None = None
        self.connection_string = connection_string

    @property
    def db_name(self):
        index = self.connection_string.rindex("/")
        return self.connection_string[index + 1:]

    def create_engine(self, echo=False, create_all=True):
        self.engine = create_engine(self.connection_string, echo=echo)
        if create_all:
            Base.metadata.create_all(self.engine)

    def create_session(self, expire_on_commit=False):
        Session = sessionmaker(bind=self.engine, autocommit=False, autoflush=False, expire_on_commit=expire_on_commit)
        self.session = Session()

    def get_session(self, expire_on_commit=False):
        Session = sessionmaker(bind=self.engine, autocommit=False, autoflush=False, expire_on_commit=expire_on_commit)
        return Session()

    def create(self, echo=False, create_all=True, expire_on_commit=False):
        self.create_engine(echo, create_all)
        self.create_session(expire_on_commit)

    def db_size(self):
        with self.engine.connect() as conn:
            sql = f"select pg_database_size('{self.db_name}')"
            res = conn.execute(text(sql))
            row = res.fetchone()
            return row[0] / 2 ** 20

