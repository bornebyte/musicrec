"""Hybrid ranking on top of pgvector: ANN/exact cosine candidates -> rerank with language, mood and co-play."""
import numpy as np
from . import config as C
from .db import arr

COLS = "id, title, artist, folder, duration, lang_name, lang_conf, mood, lang_top"


def _song(r):
    return {"id": r[0], "title": r[1], "artist": r[2], "folder": r[3], "duration": r[4],
            "language": r[5] or "unknown", "lang_conf": round(r[6] or 0, 3), "mood": r[7] or "mixed",
            "lang_guess": [f"{t['name']} {t['p']:.2f}" for t in (r[8] or [])[:3]]}


class Recommender:
    def __init__(self, pool):
        self.pool = pool

    def get(self, sid):
        with self.pool.connection() as con:
            r = con.execute(f"SELECT {COLS} FROM songs WHERE id=%s", (sid,)).fetchone()
        return _song(r) if r else None

    def search(self, q="", limit=40, language=None):
        q = (q or "").strip().lower()
        hay = "lower(title || ' ' || artist || ' ' || folder)"
        where, params = [], {"q": q, "n": limit}
        if language:
            where.append("lang_name = %(lang)s"); params["lang"] = language
        if q:
            where.append(f"({hay} LIKE '%%' || %(q)s || '%%' OR word_similarity(%(q)s, {hay}) > 0.35)")
            order = f"word_similarity(%(q)s, {hay}) DESC, title"
        else:
            order = "title"
        sql = f"SELECT {COLS} FROM songs WHERE {' AND '.join(where)} ORDER BY {order} LIMIT %(n)s"
        with self.pool.connection() as con:
            return [_song(r) for r in con.execute(sql, params)]

    def languages(self):
        with self.pool.connection() as con:
            rows = con.execute("SELECT lang_name, mood, COUNT(*) FROM songs WHERE lang_name IS NOT NULL "
                               "GROUP BY 1,2 ORDER BY 1,3 DESC").fetchall()
        out = {}
        for l, m, c in rows:
            out.setdefault(l, {"count": 0, "moods": {}})
            out[l]["count"] += c; out[l]["moods"][m] = c
        return out

    def log_play(self, session, sid, completed=False):
        with self.pool.connection() as con:
            con.execute("INSERT INTO plays(session, song_id, completed) VALUES (%s,%s,%s)", (session, sid, completed))

    def recommend(self, sid, k=20, strict_language=False, seeds=()):
        seeds = [s for s in seeds if s != sid][:3]
        with self.pool.connection() as con:
            seed = con.execute("SELECT lang_group, mood_cluster, embedding FROM songs WHERE id=%s", (sid,)).fetchone()
            if seed is None or seed[2] is None:
                return []
            group, mood_c, emb = seed
            # session-aware query vector: current song + decayed recent songs
            q = arr(emb).astype(np.float64)
            if seeds:
                got = dict(con.execute("SELECT id, embedding FROM songs WHERE id = ANY(%s) AND embedding IS NOT NULL", (seeds,)).fetchall())
                for n, s in enumerate(seeds):
                    if s in got:
                        q = q + (0.5 ** (n + 1)) * arr(got[s]).astype(np.float64)
            q = (q / (np.linalg.norm(q) + 1e-9)).astype(np.float32)

            excl = [sid] + seeds
            base = (f"SELECT {COLS}, lang_group, mood_cluster, 1 - (embedding <=> %(q)s) AS sim FROM songs "
                    "WHERE embedding IS NOT NULL AND id <> ALL(%(ex)s) {extra} ORDER BY embedding <=> %(q)s LIMIT 200")
            cands = {}
            for extra in (["AND lang_group = %(g)s"] if strict_language else ["AND lang_group = %(g)s", ""]):
                for r in con.execute(base.format(extra=extra), {"q": q, "ex": excl, "g": group}):
                    cands[r[0]] = r

            co = dict(con.execute("""
                SELECT p2.song_id, SUM(1 + p2.completed::int) FROM plays p1
                JOIN plays p2 ON p1.session = p2.session AND p2.song_id <> p1.song_id
                WHERE p1.song_id = %s GROUP BY 1""", (sid,)).fetchall())
        mx = max(co.values()) if co else 0
        out = []
        for r in cands.values():
            sim = float(r[-1]); same_lang = r[9] == group
            score = sim + C.LANG_BONUS * same_lang + C.MOOD_BONUS * (r[10] == mood_c)
            if mx and r[0] in co:
                score += C.COPLAY_WEIGHT * np.log1p(co[r[0]]) / np.log1p(mx)
            out.append(dict(_song(r), score=round(score, 4), similarity=round(sim, 4)))
        out.sort(key=lambda x: -x["score"])
        return out[:k]
