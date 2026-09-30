import sys
import time
import psycopg
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from . import config as C

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS songs (
    id           TEXT PRIMARY KEY,
    path         TEXT UNIQUE NOT NULL,
    folder       TEXT NOT NULL DEFAULT '',
    title        TEXT NOT NULL,
    artist       TEXT NOT NULL DEFAULT '',
    album        TEXT NOT NULL DEFAULT '',
    tag_genre    TEXT NOT NULL DEFAULT '',
    duration     REAL NOT NULL DEFAULT 0,
    size         BIGINT NOT NULL DEFAULT 0,
    mtime        BIGINT NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'new',      -- new | ok | failed
    error        TEXT,
    raw_features vector({C.FEATURE_DIM}),           -- MFCC/chroma/rhythm/energy statistics
    lang_emb     vector({C.LANG_DIM}),              -- ECAPA language embedding (VoxLingua107)
    lang_top     JSONB,                             -- top-5 detected languages with probabilities
    embedding    vector({C.FEATURE_DIM}),           -- normalised, weighted similarity space
    lang_group   INT,
    lang_name    TEXT,
    lang_conf    REAL,
    mood         TEXT,
    mood_cluster INT,
    arousal      REAL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS songs_embedding_hnsw ON songs USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS songs_lang_group_idx ON songs (lang_group);

CREATE TABLE IF NOT EXISTS plays (
    id        BIGSERIAL PRIMARY KEY,
    session   TEXT NOT NULL,
    song_id   TEXT NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
    completed BOOLEAN NOT NULL DEFAULT FALSE,
    ts        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS plays_session_idx ON plays (session);
CREATE INDEX IF NOT EXISTS plays_song_idx ON plays (song_id);

CREATE TABLE IF NOT EXISTS model_state (
    key        TEXT PRIMARY KEY,
    value      JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def arr(v):
    """pgvector returns numpy arrays or Vector objects depending on driver version."""
    import numpy as np
    if v is None:
        return None
    if hasattr(v, "to_numpy"):
        return v.to_numpy()
    return np.asarray(v)


def init(retries: int = 60):
    for _ in range(retries):
        try:
            with psycopg.connect(C.DATABASE_URL, autocommit=True) as con:
                con.execute("SELECT pg_advisory_lock(424242)")
                con.execute(SCHEMA)
                con.execute("SELECT pg_advisory_unlock(424242)")
            return
        except psycopg.OperationalError:
            time.sleep(1)
    sys.exit("database not reachable")


def connect():
    con = psycopg.connect(C.DATABASE_URL)
    register_vector(con)
    return con


def make_pool() -> ConnectionPool:
    return ConnectionPool(C.DATABASE_URL, min_size=1, max_size=8, open=True,
                          kwargs={"autocommit": True},
                          configure=lambda c: register_vector(c))
