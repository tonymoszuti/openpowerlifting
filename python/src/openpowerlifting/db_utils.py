from langchain_community.utilities.sql_database import SQLDatabase
from openpowerlifting.config import settings

# Database Configuration - fill in xxxx with appropriate keys
port = settings.database_port
password = settings.database_password
user = settings.database_user
host = settings.database_host
dbname = settings.database_name

db = SQLDatabase.from_uri(
    settings.database_url,
    include_tables=settings.database_tables,
    sample_rows_in_table_info=1,
)

