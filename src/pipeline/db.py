"""
db.py

Database connectivity utilities.
Supports either a single env `SQL_CONN_STR` or individual vars:
  SQL_SERVER, SQL_DATABASE, SQL_USERNAME, SQL_PASSWORD, optional SQL_DRIVER.
"""

from __future__ import annotations

import os
import logging
import pyodbc
import dotenv

# Ensure .env is loaded before accessing global variables, allowing overrides for local dev
dotenv.load_dotenv(dotenv.find_dotenv(), override=True)

logger = logging.getLogger(__name__)

SQL_ENABLED = True
SQL_ENCRYPT = os.getenv("SQL_ENCRYPT", "true").lower() == "true"
SQL_TRUST_CERT = os.getenv("SQL_TRUST_CERT", "false").lower() == "true"
SQL_TIMEOUT = int(os.getenv("SQL_CONN_TIMEOUT", os.getenv("SQL_CONNECTION_TIMEOUT", "30")))
SQL_APPNAME = os.getenv("SQL_APPNAME", "bookkeeper-api")

_DEF_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
_DEF_SERVER = os.getenv("SQL_SERVER")
_DEF_DB = os.getenv("SQL_DATABASE")
_DEF_UID = os.getenv("SQL_USERNAME")
_DEF_PWD = os.getenv("SQL_PASSWORD")
_SQL_CONN_STR = os.getenv("SQL_CONN_STR")


def _parse_conn_str(conn: str) -> dict:
    parts = [p.strip() for p in conn.replace("\n", "").split(";") if p.strip()]
    kv: dict = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip().upper()] = v.strip().strip("{}")
    return kv


def _build_conn_info() -> dict:
    # Fetch variables at runtime to support dynamic updates (e.g., uvicorn reload)
    sql_conn_str = os.getenv("SQL_CONN_STR")
    def_server = os.getenv("SQL_SERVER")
    def_db = os.getenv("SQL_DATABASE")
    def_uid = os.getenv("SQL_USERNAME")
    def_pwd = os.getenv("SQL_PASSWORD")
    def_driver = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")

    conn = None
    if sql_conn_str:
        conn = "".join(line.strip() for line in sql_conn_str.splitlines())
        logger.info("Using SQL_CONN_STR env for database connection.")
        kv = _parse_conn_str(conn)
        server = kv.get("SERVER") or kv.get("DATA SOURCE")
        database = kv.get("DATABASE") or kv.get("INITIAL CATALOG")
        uid = kv.get("UID") or kv.get("USER ID")
        pwd = kv.get("PWD") or kv.get("PASSWORD")
        driver = kv.get("DRIVER") or def_driver
        encrypt = kv.get("ENCRYPT", "yes").lower() in ("yes", "true", "1")
        trust = kv.get("TRUSTSERVERCERTIFICATE", "no").lower() in ("yes", "true", "1")
        logger.info(f"Using SQL_CONN_STR for database connection (Server: {server}, DB: {database}, Driver: {driver})")
    else:
        missing = [k for k, v in {
            'SQL_SERVER': def_server,
            'SQL_DATABASE': def_db,
            'SQL_USERNAME': def_uid,
            'SQL_PASSWORD': def_pwd,
        }.items() if not v]
        if missing:
            raise RuntimeError(f"Missing SQL env vars: {', '.join(missing)}")
        server, database, uid, pwd, driver = def_server, def_db, def_uid, def_pwd, def_driver
        encrypt, trust = SQL_ENCRYPT, SQL_TRUST_CERT
        conn = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={uid};"
            f"PWD={pwd};"
            f"Encrypt={'yes' if encrypt else 'no'};"
            f"TrustServerCertificate={'yes' if trust else 'no'};"
            f"Connection Timeout={SQL_TIMEOUT};"
            f"APP={SQL_APPNAME};"
        )
    return {
        "conn_str": conn,
        "server": server,
        "database": database,
        "uid": uid,
        "pwd": pwd,
        "driver": driver,
        "encrypt": encrypt,
        "trust": trust,
    }


class _RowAdapter:
    def __init__(self, values, columns):
        self._values = tuple(values)
        self._columns = [c[0] for c in columns] if columns else []
        self._map = {name: idx for idx, name in enumerate(self._columns)}

    def __getattr__(self, item):
        if item in self._map:
            return self._values[self._map[item]]
        raise AttributeError(item)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[key]
        if isinstance(key, str) and key in self._map:
            return self._values[self._map[key]]
        raise KeyError(key)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class _CursorAdapter:
    def __init__(self, cur):
        self._cur = cur

    def execute(self, *args, **kwargs):
        return self._cur.execute(*args, **kwargs)

    def fetchone(self):
        row = self._cur.fetchone()
        if row is None:
            return None
        return _RowAdapter(row, getattr(self._cur, "description", []))

    def fetchall(self):
        rows = self._cur.fetchall()
        desc = getattr(self._cur, "description", [])
        return [_RowAdapter(r, desc) for r in rows]

    def __getattr__(self, item):
        return getattr(self._cur, item)


def get_sql_conn():
    """Return a live pyodbc connection."""
    if not SQL_ENABLED:
        raise RuntimeError("SQL disabled")
    info = _build_conn_info()
    logger.debug("Connecting to SQL Server via pyodbc %s / %s", info["server"], info["database"])
    return pyodbc.connect(info["conn_str"], timeout=SQL_TIMEOUT)


def ensure_schema():
    """Ensure required database schema exists (e.g. gold.ProcessorConfig)."""
    if not SQL_ENABLED:
        return
    
    sql = """
    IF NOT EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[gold].[ProcessorConfig]') AND type in (N'U'))
    BEGIN
        CREATE TABLE [gold].[ProcessorConfig](
            [ProcessorID] [int] IDENTITY(1,1) NOT NULL,
            [Name] [varchar](100) NOT NULL UNIQUE,
            [DisplayName] [varchar](100) NULL,
            [Description] [nvarchar](max) NULL,
            [SystemPrompt] [nvarchar](max) NULL,
            [UserPrompt] [nvarchar](max) NULL,
            [SchemaDefinition] [nvarchar](max) NULL,
            [Enabled] [bit] DEFAULT 1,
            [IsSystem] [bit] DEFAULT 0,
            [CreatedAt] [datetime] DEFAULT GETUTCDATE(),
            [UpdatedAt] [datetime] DEFAULT GETUTCDATE(),
            PRIMARY KEY CLUSTERED ([ProcessorID] ASC)
        );
    END

    -- Ensure dimDocument has validation columns
    IF EXISTS (SELECT * FROM sys.objects WHERE object_id = OBJECT_ID(N'[gold].[dimDocument]') AND type in (N'U'))
    BEGIN
        IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[gold].[dimDocument]') AND name = 'ValidationErrors')
        BEGIN
            ALTER TABLE [gold].[dimDocument] ADD [ValidationErrors] [nvarchar](max) NULL;
        END
        IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[gold].[dimDocument]') AND name = 'ValidationWarnings')
        BEGIN
            ALTER TABLE [gold].[dimDocument] ADD [ValidationWarnings] [nvarchar](max) NULL;
        END
        IF NOT EXISTS (SELECT * FROM sys.columns WHERE object_id = OBJECT_ID(N'[gold].[dimDocument]') AND name = 'ClassificationConfidence')
        BEGIN
            ALTER TABLE [gold].[dimDocument] ADD [ClassificationConfidence] [float] NULL;
        END
    END
    """
    conn = get_sql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            conn.commit()
        logger.info("[DB] Schema verification complete (gold.ProcessorConfig ensured)")
    except Exception as e:
        logger.error(f"[DB] Schema verification failed: {e}")
        # We don't raise here to allow app to start even if SQL is partially broken, 
        # but features relying on this table will fail later.
    finally:
        conn.close()
