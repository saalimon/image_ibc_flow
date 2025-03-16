import logger
import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine, inspect

CONFIG = dotenv_values(".env")


@logger
def gen_connection_str(conf, db='master'):
    '''
    Generate connection string for Postgresql Server
    param 
        conf: dict, configuration
        db: str, database name
    return
        str, connection string
    '''
    return "postgresql+psycopg2://{}:{}@{}:{}/{}".format(
        conf["POSTGRES_USER"],
        conf["POSTGRES_PASSWORD"],
        conf["POSTGRES_HOST"],
        conf["POSTGRES_PORT"],
        db
    )


@logger
def connect_db(db='master'):
    '''
    Connect to Postgresql Server
    param
        db: str, database name
    return
        engine, connection to database
    '''
    print("Connecting to DB")
    connection_uri = gen_connection_str(CONFIG, db)
    engine = create_engine(connection_uri, pool_pre_ping=True)
    engine.connect()
    return engine
