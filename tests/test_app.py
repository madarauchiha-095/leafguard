import asyncio
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import server


class AppTests(unittest.TestCase):
    def test_health_endpoint_returns_service_status(self):
        response = asyncio.run(server.health(None))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.body), {
            "status": "ok",
            "service": "LeafGuard",
        })

    def test_database_initialization_creates_app_tables(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_directory = Path(temporary_directory)
            database = state_directory / "users.sqlite3"
            with (
                patch.object(server, "STATE_DIR", state_directory),
                patch.object(server, "DATABASE", database),
            ):
                server._initialize_database()

            connection = sqlite3.connect(database)
            try:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
            finally:
                connection.close()

        self.assertTrue({"users", "analysis_runs", "analysis_results"} <= tables)


if __name__ == "__main__":
    unittest.main()
