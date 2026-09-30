import mimetypes
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config as C
from . import db
from .recommender import Recommender

db.init()
pool = db.make_pool()
rec = Recommender(pool)
app = FastAPI(title="MusicRec")


class Play(BaseModel):
    session: str
    song: str
    completed: bool = False


@app.get("/api/search")
def search(q: str = "", language: str | None = None, limit: int = 40):
    return rec.search(q, limit, language or None)


@app.get("/api/similar/{sid}")
def similar(sid: str, k: int = 25, strict_language: bool = False, seeds: str = Query("")):
    if rec.get(sid) is None:
        raise HTTPException(404)
    return rec.recommend(sid, k, strict_language, [s for s in seeds.split(",") if s])


@app.get("/api/languages")
def languages():
    return rec.languages()


@app.post("/api/play")
def play(p: Play):
    rec.log_play(p.session, p.song, p.completed)
    return {"ok": True}


@app.get("/api/stream/{sid}")
def stream(sid: str):
    with pool.connection() as con:
        r = con.execute("SELECT path FROM songs WHERE id=%s", (sid,)).fetchone()
    if not r:
        raise HTTPException(404)
    p = C.MUSIC_DIR / r[0]
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "audio/mpeg")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
