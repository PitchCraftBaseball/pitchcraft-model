import os
from typing import Iterator, Optional
from dotenv import load_dotenv
from contextlib import contextmanager
import psycopg2.pool
from psycopg2.extensions import cursor as Cursor
from psycopg2 import sql
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
    
def query_table_for_features(table_name: str, features: list[str]) -> Cursor:
    query = sql.SQL(
        "select {fields} from {table} "
        "where {date_col} >= date_trunc('year', current_date) - interval '1 year' "
        "and {date_col} < date_trunc('year', current_date);"
    ).format(
        fields=sql.SQL(',').join([
            sql.Identifier(x) for x in features
        ]),
        table=sql.Identifier(table_name),
        date_col=sql.Identifier("game_date")
    )
    # run query 
    with get_read_cursor() as cursor: 
        cursor.execute(query)
        print(f"Query completed")
        return cursor.fetchall()


def find_table_for_column(schema: str, column: str) -> Optional[str]:
    # Return the first table in the schema that contains the requested column.
    sql = """
        SELECT c.relname
        FROM pg_attribute a
        JOIN pg_class c ON c.oid = a.attrelid
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = %s
          AND c.relkind IN ('r','p','v','m','f')
          AND a.attname = %s
          AND a.attnum > 0
          AND NOT a.attisdropped
        LIMIT 1
    """
    with get_read_cursor() as cursor:
        cursor.execute(sql, (schema, column))
        row = cursor.fetchone()
        return row[0] if row else None
