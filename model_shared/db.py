import os
from pathlib import Path
from typing import Iterator, Optional
from dotenv import load_dotenv
from contextlib import contextmanager
import psycopg2.pool
from psycopg2.extensions import cursor as Cursor
from psycopg2 import sql
from .logger import logger
import pandas as pd


load_dotenv()

_repo_root = Path(__file__).resolve().parents[1]
_cert_path = os.getenv("DB_RDS_CERT_PATH")
if _cert_path:
    cert_path = Path(_cert_path)
    if not cert_path.is_absolute():
        _cert_path = str(_repo_root / cert_path)

pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    dbname=os.getenv("DB_DATABASE"),
    sslmode="verify-full",
    sslrootcert=_cert_path
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
    
def query_historical_pitches_by_year(table_name: str, features: list[str], start_year: int = 2015, end_year: int = 2025) -> pd.DataFrame:
    dfs = []
    
    for year in range(start_year, end_year + 1):
        query = sql.SQL(
            "SELECT {fields} FROM {table} "
            "WHERE {date_col} >= %s AND {date_col} < %s;"
        ).format(
            fields=sql.SQL(', ').join([sql.Identifier(f) for f in features]),
            table=sql.Identifier(table_name),
            date_col=sql.Identifier("game_date"),
        )
        
        with get_read_cursor() as cursor:
            cursor.execute(query, (f"{year}-01-01", f"{year + 1}-01-01"))
            rows = cursor.fetchall()
            year_df = pd.DataFrame(rows, columns=features)
            dfs.append(year_df)
            print(f"Fetched {len(year_df):,} rows for {year}")
    
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows fetched: {len(full_df):,}")
    return full_df


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
