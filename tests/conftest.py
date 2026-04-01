import os
import subprocess

import pytest
from sqlalchemy import create_engine
from testcontainers.postgres import PostgresContainer

@pytest.fixture(scope="session")
def setup(request):
    try:
        postgres = PostgresContainer("postgres:16-alpine")
        postgres.start()
    except Exception as e:
        pytest.skip(f"Docker/Postgres testcontainer unavailable: {e}")

    def remove_container():
        postgres.stop()

    request.addfinalizer(remove_container)
    os.environ["DB_URL_TEST"] = postgres.get_connection_url()


@pytest.fixture(scope="session")
def apply_migrations(setup):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    alembic_command = [sys.executable, "-m", "alembic", "upgrade", "head"]
    subprocess.run(alembic_command, check=True, cwd=project_root)


@pytest.fixture(scope="module")
def db_engine(apply_migrations):
    engine = create_engine(os.environ["DB_URL_TEST"])
    yield engine
    engine.dispose()
