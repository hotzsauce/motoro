"""
expose a SQL interface
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from trino.auth import BasicAuthentication



class SqlInterface(object):

    def __init__(
        self,
        user: str = "trino_read",
        password: str = "SecretTrinoModoPassword123!",
        host: str = "trino-prod.modoint.com",
        port: int = 443,
        catalog: str = "iceberg",
        echo: bool = True,
    ):
        self.auth = BasicAuthentication(user, password)
        self.engine = create_engine(
            f"trino://{user}@{host}:{port}/{catalog}",
            connect_args={
                "http_scheme": "https",
                "auth": self.auth,
            },
            echo=echo,
        )
        self.connection = self.engine.connect()

    def read_sql(self, query: str) -> pd.DataFrame:
        return pd.read_sql(query, self.connection)
