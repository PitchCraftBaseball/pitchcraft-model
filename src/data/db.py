import os
from typing import Iterator
from dotenv import load_dotenv
from contextlib import contextmanager
import psycopg2.pool
from psycopg2.extensions import cursor as Cursor
from src.utils.logger import logger


load_dotenv()

pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_DATABASE"),
    sslmode="verify-full",
    sslrootcert=os.getenv("DB_RDS_CERT_PATH")
)

@contextmanager
def get_read_cursor() -> Iterator[Cursor]:
    conn = pool.getconn()
    try:
        conn.set_client_encoding("UTF8")
        cursor = conn.cursor()
        yield cursor
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        cursor.close()
        pool.putconn(conn)
