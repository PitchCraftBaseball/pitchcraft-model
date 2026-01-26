import os
from typing import Iterator, Optional
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

# TODO: was thinking of using this to figure out what DB to query if we wanted to handle everything from the backend (and programatically)
def column_exists(schema: str, column: str, table: Optional[str] = None) -> bool:
    # Check whether a column exists using Postgres catalogs (faster than information_schema).
    # SQL breakdown:
    # - SELECT 1: Only need existence, not data.
    # - FROM pg_attribute a: Column metadata (attributes).
    # - JOIN pg_class c ON c.oid = a.attrelid: Link columns to their tables/views.
    # - JOIN pg_namespace n ON n.oid = c.relnamespace: Link tables to schemas.
    # - WHERE n.nspname = %s: Restrict to the requested schema.
    # - AND c.relname = %s: (table-specific branch) Restrict to the requested table.
    # - AND c.relkind IN ('r','p','v','m','f'): Limit to tables, partitions, views, matviews, foreign tables.
    # - AND a.attname = %s: Match the column name.
    # - AND a.attnum > 0: Exclude system columns.
    # - AND NOT a.attisdropped: Exclude dropped columns.
    # - LIMIT 1: Stop on the first match.
    if table is None:
        sql = """
            SELECT 1
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
        params = (schema, column)
    else:
        sql = """
            SELECT 1
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relname = %s
              AND c.relkind IN ('r','p','v','m','f')
              AND a.attname = %s
              AND a.attnum > 0
              AND NOT a.attisdropped
            LIMIT 1
        """
        params = (schema, table, column)
    with get_read_cursor() as cursor:
        cursor.execute(sql, params)
        return cursor.fetchone() is not None


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
