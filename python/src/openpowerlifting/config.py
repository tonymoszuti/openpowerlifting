from pathlib import Path

from dotenv import load_dotenv
from dynaconf.base import LazySettings

env_path = Path("python") / ".env"
load_dotenv(dotenv_path=env_path, verbose=True)

settings = LazySettings(
    envvar_prefix="SQL_AGENT",
    environments=["development", "production"],
    loaders=['dynaconf.loaders.env_loader'],
    settings_files=['python/settings.toml',
                    'python/secrets.toml'],
)
settings.configure()

settings.database_url = (
    f"postgresql+psycopg2://{settings.database_user}:"
    f"{settings.database_password}@{settings.database_host}:"
    f"{settings.database_port}/{settings.database_name}"
)
