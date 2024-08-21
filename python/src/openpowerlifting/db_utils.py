import psycopg2
import logging
from langchain_community.utilities.sql_database import SQLDatabase

# Database Configuration - fill in xxxx with appropriate keys
port = "xxxx"
password = "xxxx"
user = "xxxx"
host = "postgres-db"
dbname = "openpowerlifting"

def create_postgres_connection():
    """
    Creates a PostgreSQL database connection.
    """
    try:
        connection = psycopg2.connect(
            database=dbname, user=user, password=password, host=host, port=port
        )
        return connection
    except psycopg2.OperationalError as e:
        logging.error("Database connection failed: %s", e)
        return None


url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
TABLE_NAME = "powerlifting_results_final"

db = SQLDatabase.from_uri(
    url,
    include_tables=[TABLE_NAME],
    sample_rows_in_table_info=1,
)

