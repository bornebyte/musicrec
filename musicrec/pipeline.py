"""scan -> extract -> build, all state stored in Postgres/pgvector."""
import hashlib
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import multiprocessing as mp

import numpy as np
from mutagen import File as MFile
from psycopg.types.json import Jsonb
from sklearn.cluster import KMeans
from tqdm import tqdm

from . import config as C
from . import db
from .audio import probe_duration
from .features import extract_features, block_slices, BLOCK_SIZES


def song_id(rel: str) -> str:
    return hashlib.sha1(rel.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ scan
def _tags(p):
    try:
        t = MFile(str(p), easy=True)
        g = lambda k: ((t.get(k) or [""])[0] if t is not None else "") or ""
        return g("title"), g("artist"), g("album"), g("genre")
    except Exception:
        return "", "", "", ""


def scan():
    files = sorted(p for p in C.MUSIC_DIR.rglob("*") if p.is_file() and p.suffix.lower() in C.AUDIO_EXT)
    if not files:
        sys.exit(f"no audio files under {C.MUSIC_DIR} (is the folder mounted?)")
    with db.connect() as con:
        existing = {r[0]: (r[1], r[2]) for r in con.execute("SELECT path,size,mtime FROM songs")}
        seen, added = set(), 0
        for p in tqdm(files, desc="scan"):
            rel = p.relative_to(C.MUSIC_DIR).as_posix()
            seen.add(rel)
            st = p.stat()
            sig = (st.st_size, int(st.st_mtime))
            if existing.get(rel) == sig:
                continue
            title, artist, album, genre = _tags(p)
            folder = rel.split("/")[0] if "/" in rel else ""
            con.execute("""
                INSERT INTO songs(id,path,folder,title,artist,album,tag_genre,duration,size,mtime)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (path) DO UPDATE SET
                  title=EXCLUDED.title, artist=EXCLUDED.artist, album=EXCLUDED.album, tag_genre=EXCLUDED.tag_genre,
                  duration=EXCLUDED.duration, size=EXCLUDED.size, mtime=EXCLUDED.mtime,
                  raw_features=NULL, lang_emb=NULL, lang_top=NULL, embedding=NULL,
                  status='new', error=NULL, updated_at=now()""",
                        (song_id(rel), rel, folder, title or p.stem, artist, album, genre,
                         probe_duration(p), st.st_size, sig[1]))
            added += 1
        gone = set(existing) - seen
        if gone:
            con.execute("DELETE FROM songs WHERE path = ANY(%s)", (list(gone),))
    print(f"scan: {len(files)} files, {added} new/changed, {len(gone)} removed")


# ------------------------------------------------------------------ extract (parallel)
_lid = None


def _analyse(sid, rel):
    global _lid
    path = C.MUSIC_DIR / rel
    try:
        dur = probe_duration(path)
        feats = extract_features(path, dur)
        lid = None
        if C.LANGID_ENABLED and _lid is not False:
            if _lid is None:
                try:
                    from .langid import LangID
                    _lid = LangID()
                except Exception as e:
                    print(f"[warn] language-ID model unavailable ({e}); falling back to folder hints", file=sys.stderr)
                    _lid = False
            if _lid:
                lid = _lid.analyse(path, dur)
        return sid, feats, lid, None
    except Exception as e:
        return sid, None, None, f"{type(e).__name__}: {e}"[:500]


def extract(retry=False, rebuild=False):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    with db.connect() as con:
        if rebuild:
            con.execute("UPDATE songs SET raw_features=NULL, lang_emb=NULL, lang_top=NULL, embedding=NULL, status='new'")
        q = "SELECT id,path FROM songs WHERE raw_features IS NULL" + ("" if retry else " AND status <> 'failed'")
        todo = con.execute(q).fetchall()
    if not todo:
        print("extract: nothing to do"); return
    print(f"extract: {len(todo)} songs, {C.WORKERS} workers (first run downloads the language model)")

    pending = dict(todo)
    bar = tqdm(total=len(todo), desc="features")
    for _round in range(4):                                   # survive worker crashes
        if not pending:
            break
        try:
            with ProcessPoolExecutor(C.WORKERS, mp_context=mp.get_context("spawn")) as ex, db.connect() as con:
                futs = [ex.submit(_analyse, sid, rel) for sid, rel in pending.items()]
                for i, f in enumerate(as_completed(futs), 1):
                    sid, feats, lid, err = f.result()
                    if err:
                        con.execute("UPDATE songs SET status='failed', error=%s, updated_at=now() WHERE id=%s", (err, sid))
                        tqdm.write(f"  failed {pending[sid]}: {err}")
                    else:
                        con.execute(
                            "UPDATE songs SET raw_features=%s, lang_emb=%s, lang_top=%s, status='ok', error=NULL, updated_at=now() WHERE id=%s",
                            (feats, lid["emb"] if lid else None, Jsonb(lid["top"]) if lid else None, sid))
                    pending.pop(sid, None)
                    bar.update(1)
                    if i % 25 == 0:
                        con.commit()
                con.commit()
        except BrokenProcessPool:
            tqdm.write("  a worker crashed; restarting pool for the remaining songs")
    bar.close()
    if pending:
        print(f"extract: {len(pending)} songs left unprocessed (worker crashes) - re-run `extract`")


# ------------------------------------------------------------------ build (corpus-level)
def group_languages(LE, tops, folders):
    n = len(folders)
    stems = [C.folder_stem(f) for f in folders]
    has = np.array([e is not None for e in LE])
    if has.sum() < max(10, 2 * C.LANG_MIN_GROUP):
        print("[warn] not enough language-ID data; grouping by folder name instead")
        uniq = sorted(set(stems)); gid = {s: i for i, s in enumerate(uniq)}
        return np.array([gid[s] for s in stems]), list(stems), np.zeros(n)

    idx = np.where(has)[0]
    Xl = np.vstack([LE[i] for i in idx]).astype(np.float64)
    Xl /= np.linalg.norm(Xl, axis=1, keepdims=True) + 1e-9
    top1 = [tops[i][0]["name"] if tops[i] else "unknown" for i in idx]
    cnt = Counter(top1)
    big = [l for l, c in cnt.items() if c >= C.LANG_MIN_GROUP]
    k = C.LANG_K or int(np.clip(len(big), 2, 10))
    k = max(1, min(k, len(idx) // max(C.LANG_MIN_GROUP, 1)))
    km = KMeans(k, n_init=10, random_state=0).fit(Xl)

    base, hint = {}, {}
    for c in range(k):
        mem = np.where(km.labels_ == c)[0]
        base[c] = Counter(top1[m] for m in mem).most_common(1)[0][0]
        hint[c] = Counter(stems[idx[m]] for m in mem).most_common(1)[0][0]
    used = Counter(base.values())
    label = {}
    for c in range(k):
        if used[base[c]] > 1:
            tag = hint[c] if hint[c] != base[c].lower() else f"group {c + 1}"
            lab = f"{base[c]} ({tag})"
            if lab in label.values():
                lab += f" #{c + 1}"
        else:
            lab = base[c]
        label[c] = lab

    gid = np.full(n, -1); conf = np.zeros(n)
    for pos, i in enumerate(idx):
        c = int(km.labels_[pos]); gid[i] = c
        conf[i] = next((t["p"] for t in (tops[i] or []) if t["name"] == base[c]), 0.0)
    stem_mode = {}
    for s in set(stems):
        cs = Counter(int(gid[i]) for i in idx if stems[i] == s)
        if cs:
            stem_mode[s] = cs.most_common(1)[0][0]
    names = dict(label); names[k] = "unknown"
    for i in np.where(~has)[0]:
        gid[i] = stem_mode.get(stems[i], k)
    return gid, [names[int(g)] for g in gid], conf


def build():
    with db.connect() as con:
        rows = con.execute("SELECT id, raw_features, lang_emb, lang_top, folder FROM songs "
                           "WHERE raw_features IS NOT NULL ORDER BY id").fetchall()
        if len(rows) < 5:
            sys.exit("build: need at least 5 extracted songs")
        ids = [r[0] for r in rows]
        X = np.vstack([db.arr(r[1]) for r in rows]).astype(np.float64)
        LE = [db.arr(r[2]) for r in rows]; tops = [r[3] for r in rows]; folders = [r[4] for r in rows]
        n = len(ids)

        # 1) similarity space: robust z-score -> block equalisation -> centre -> L2
        med = np.median(X, 0)
        iqr = np.subtract(*np.percentile(X, [75, 25], axis=0)) + 1e-9
        Z = np.clip((X - med) / (iqr / 1.349), -4, 4)
        W = np.empty_like(Z)
        for b, sl in block_slices().items():
            W[:, sl] = Z[:, sl] * C.BLOCK_WEIGHTS[b] / np.sqrt(BLOCK_SIZES[b])
        W -= W.mean(0)
        E = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
        sl = block_slices(); r0, e0 = sl["rhythm"].start, sl["energy"].start
        arousal = (Z[:, r0] + Z[:, r0 + 1] + Z[:, e0] + Z[:, e0 + 2]) / 4     # tempo, onset, rms, centroid

        # 2) language groups (per song)
        gid, lname, lconf = group_languages(LE, tops, folders)

        # 3) mood clusters inside each language group
        mood = np.array(["mixed"] * n, dtype=object); mood_id = np.zeros(n, dtype=int)
        names = ["calm / soft", "mellow / emotional", "groovy / mid-tempo", "upbeat", "energetic / party"]
        next_id = 1
        for g in np.unique(gid):
            idx = np.where(gid == g)[0]
            k = int(np.clip(len(idx) // 12, 1, C.MAX_MOOD_CLUSTERS))
            if k < 2:
                mood_id[idx] = next_id; next_id += 1; continue
            km = KMeans(k, n_init=10, random_state=0).fit(E[idx])
            order = np.argsort([arousal[idx][km.labels_ == c].mean() for c in range(k)])
            rank = {int(c): r for r, c in enumerate(order)}
            for c in range(k):
                m = idx[km.labels_ == c]
                mood[m] = names[int(round(rank[c] / (k - 1) * (len(names) - 1)))]
                mood_id[m] = next_id; next_id += 1

        con.cursor().executemany(
            "UPDATE songs SET embedding=%s, lang_group=%s, lang_name=%s, lang_conf=%s, mood=%s, mood_cluster=%s, arousal=%s, "
            "updated_at=now() WHERE id=%s",
            [(E[i].astype(np.float32), int(gid[i]), lname[i], float(lconf[i]), str(mood[i]), int(mood_id[i]),
              float(arousal[i]), ids[i]) for i in range(n)])
        con.execute("INSERT INTO model_state(key,value) VALUES ('space', %s) ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=now()",
                    (Jsonb({"median": med.tolist(), "iqr": iqr.tolist(), "weights": C.BLOCK_WEIGHTS, "n": n}),))
    print(f"build: {n} songs -> {len(set(lname))} language groups: {sorted(set(lname))}")


# ------------------------------------------------------------------ reports
def report():
    with db.connect() as con:
        print("\n== language groups ==")
        for r in con.execute("SELECT lang_name, COUNT(*), ROUND(AVG(lang_conf)::numeric,2) FROM songs "
                             "WHERE lang_name IS NOT NULL GROUP BY 1 ORDER BY 2 DESC"):
            print(f"  {r[0]:32s} {r[1]:5d} songs   mean LID confidence {r[2]}")
        print("\n== folder x detected language (mixed folders show up here) ==")
        for r in con.execute("SELECT folder, lang_name, COUNT(*) FROM songs WHERE lang_name IS NOT NULL "
                             "GROUP BY 1,2 ORDER BY 1, 3 DESC"):
            print(f"  {r[0]:18s} {r[1]:32s} {r[2]:5d}")
        print("\n== language x mood ==")
        for r in con.execute("SELECT lang_name, mood, COUNT(*) FROM songs WHERE lang_name IS NOT NULL "
                             "GROUP BY 1,2 ORDER BY 1, 3 DESC"):
            print(f"  {r[0]:32s} {r[1]:22s} {r[2]:5d}")
        bad = con.execute("SELECT path, error FROM songs WHERE status='failed'").fetchall()
        if bad:
            print(f"\n== {len(bad)} failed files ==")
            for p, e in bad[:30]:
                print(f"  {p}: {e}")


def evaluate(sample=200):
    from .recommender import Recommender
    pool = db.make_pool(); rec = Recommender(pool)
    with pool.connection() as con:
        ids = [r[0] for r in con.execute("SELECT id FROM songs WHERE embedding IS NOT NULL")]
    rng = np.random.default_rng(0); pick = rng.choice(ids, min(sample, len(ids)), replace=False)
    lang_hit, mood_hit = [], []
    for sid in pick:
        seed = rec.get(sid); out = rec.recommend(sid, 10)
        lang_hit.append(np.mean([o["language"] == seed["language"] for o in out]))
        mood_hit.append(np.mean([o["mood"] == seed["mood"] for o in out]))
    print(f"language-consistency@10 = {np.mean(lang_hit):.3f}   mood-consistency@10 = {np.mean(mood_hit):.3f}")
    s = rec.get(str(rng.choice(ids)))
    print(f"\nseed: {s['title']} [{s['language']} | {s['mood']}]")
    for o in rec.recommend(s["id"], 10):
        print(f"  {o['score']:.3f}  {o['title'][:40]:40s} {o['language']:22s} {o['mood']}")
    pool.close()
