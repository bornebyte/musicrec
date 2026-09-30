# MusicRec v2 — Docker + Postgres/pgvector

    docker compose up -d --build db app            # DB + API (http://localhost:8000)
    docker compose run --rm app python -m musicrec all   # scan -> extract -> build -> report
    # optional: MUSIC_DIR=/path/to/songs WORKERS=6 docker compose run --rm app python -m musicrec all

Other commands (`docker compose run --rm app python -m musicrec <cmd>`):
`scan` · `extract [--retry|--rebuild]` · `build` (recluster/re-embed, seconds) · `report` · `eval`

Inspect data: `psql -h 127.0.0.1 -p 5433 -U musicrec musicrec` (password `musicrec`), tables `songs`, `plays`, `model_state`.

Language is detected **per song** (SpeechBrain VoxLingua107 ECAPA, 3 windows/song), then songs are clustered on the
language embedding; groups are named by majority detected language. Folder names are never used as labels
(only as a tie-break hint for naming duplicates). `report` shows folder x language so mixed folders are visible.
Force more groups with `LANG_K=6` (e.g. to try to split Bhojpuri from Hindi — VoxLingua107 has no Bhojpuri class).
