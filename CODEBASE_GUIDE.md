# MusicRec codebase guide

This is a beginner-oriented map of the repository. It explains the purpose of
each file, the order in which the program runs, the important values produced
by each stage, and the database/vector concepts used by the project.

MusicRec is a local music-library recommender:

    audio files on disk
            |
            v
    scan metadata and file information
            |
            v
    extract 81 audio features + optional 256-d language data
            |
            v
    build normalized similarity vectors, language groups, and mood groups
            |
            v
    PostgreSQL + pgvector <-------- FastAPI <-------- browser UI
                                      |
                                      +-- search
                                      +-- stream audio
                                      +-- recommendations
                                      +-- play history

The important distinction is that this does not store audio in PostgreSQL.
Audio remains on the mounted filesystem. PostgreSQL stores metadata, analysis
results, vectors, grouping labels, and listening events.

## Recommended reading order

Read the project in this order instead of starting with the large HTML file:

1. README.md — intended commands and high-level behavior.
2. musicrec/config.py — constants that control every stage.
3. musicrec/__main__.py — command-line entry point and top-level order.
4. musicrec/db.py — tables, vector columns, indexes, connections, and pooling.
5. musicrec/audio.py — how a file becomes a NumPy waveform.
6. musicrec/features.py — how the waveform becomes an 81-number audio vector.
7. musicrec/langid.py — how language identification produces a 256-number
   vector and language probabilities.
8. musicrec/pipeline.py — scan, extract, build, and report; this is the main
   data pipeline.
9. musicrec/recommender.py — SQL search and ranking.
10. musicrec/api.py — HTTP endpoints that expose the recommender.
11. musicrec/static/index.html — browser behavior and playback UI. Most of its
    CSS is presentation; the JavaScript near the end is the important part.

Read the deployment files after the Python flow:

- requirements.txt lists Python packages.
- Dockerfile builds the image and installs native audio dependencies.
- docker-compose.yml starts PostgreSQL and the API, mounts the music directory,
  and persists database/model data.
- update.sh automates scan -> extract -> build -> report.

## How the application starts

There are two entry points.

### Command-line pipeline

    python -m musicrec scan
    python -m musicrec extract --retry
    python -m musicrec build
    python -m musicrec report

Python sees musicrec as a package and executes __main__.py. argparse reads the
command. Then line 15 calls db.init() before any pipeline command runs, so the
schema is created if necessary. The selected command calls one function in
pipeline.py.

python -m musicrec all calls all four stages in sequence:

    pipeline.scan()
    pipeline.extract(...)
    pipeline.build()
    pipeline.report()

The values returned by these pipeline functions are generally None; their
useful output is database updates and printed progress. Individual helpers
return the data that the next stage needs.

### Web API

Uvicorn loads musicrec.api:app, which executes module-level setup in api.py:

1. db.init() waits for PostgreSQL and creates the schema.
2. db.make_pool() creates a connection pool.
3. Recommender(pool) creates an object that uses the pool.
4. FastAPI registers HTTP routes.
5. The / route serves static/index.html.

The API process does not run scan, extract, or build automatically. Run the
pipeline separately first.

## Python package details

### Why is __init__.py empty?

An empty musicrec/__init__.py still has a purpose: it marks the directory as
the musicrec package and makes imports such as from . import db work
consistently. It is also the natural place for package metadata or public
exports later, but this project does not need either.

Modern Python can sometimes import a directory without __init__.py as a
namespace package. Keeping the empty file is clearer, works with older tooling,
and explicitly says “this directory is a Python package.” Importing it produces
no visible value and runs no application setup.

### Relative imports

In pipeline.py, from . import config as C means “import config.py from this
same package and call it C.” The dot works because the code is run as a package,
normally with python -m musicrec ...

### Context managers

Code such as:

    with db.connect() as con:
        rows = con.execute("SELECT ...").fetchall()

automatically closes the connection when the block ends and handles the
transaction according to psycopg's connection context behavior. The same idea
is used for pooled connections. It prevents forgotten connections from
remaining open.

### None, lists, dictionaries, and NumPy arrays

- None means “there is no value,” for example no language embedding.
- A list is an ordered Python collection, for example a list of file paths.
- A dictionary maps names to values, for example {"title": "...", "p": 0.8}.
- A NumPy array is a compact numerical container. Its shape describes its
  dimensions: (81,) is one vector with 81 values, while (20, frames) is a
  matrix with 20 rows of frame-by-frame measurements.
- np.vstack([...]) stacks vectors into rows. If there are 100 songs, a list of
  81-value vectors becomes an array with shape (100, 81).

## Configuration: musicrec/config.py

This module is imported by almost every backend module. It reads environment
variables once and defines constants:

| Setting | Meaning | Default |
|---|---|---:|
| DATABASE_URL | PostgreSQL connection URL | local PostgreSQL on port 5433 |
| MUSIC_DIR | root directory containing songs | /music |
| MODEL_DIR | cache directory for SpeechBrain | /models |
| AUDIO_EXT | suffixes accepted by scan | mp3/flac/wav/m4a/ogg/opus/aac |
| SR | sample rate for audio features | 22,050 Hz |
| CLIP_SECONDS | middle clip used for features | 45 seconds |
| LID_SR | sample rate for language ID | 16,000 Hz |
| LID_WINDOW | each language-ID window | 8 seconds |
| LID_WINDOWS | maximum windows per song | 3 |
| WORKERS | extraction processes | 4 |
| LANGID | set to 0 to disable language identification | enabled |
| LANG_K | language cluster count; 0 means automatic | automatic |
| LANG_MIN_GROUP | minimum frequent-language group size | 8 |

FEATURE_DIM = 81 and LANG_DIM = 256 are schema contracts. Vectors written into
the corresponding PostgreSQL columns must have exactly those lengths.

folder_stem("hindi 2") returns "hindi". It is only a fallback/naming hint;
language identification is intended to be the primary language source.

## Database concepts: musicrec/db.py

### PostgreSQL is still relational

The project uses normal relational features:

- rows and columns;
- primary keys (songs.id, plays.id, model_state.key);
- unique constraints (songs.path);
- foreign keys (plays.song_id -> songs.id);
- SELECT, INSERT, UPDATE, and DELETE;
- indexes and transactions.

The vector feature is an extension to PostgreSQL, not a replacement for it.
CREATE EXTENSION vector adds the vector(N) type and vector operators.

### The songs table

There is one row per discovered audio file:

| Column group | Columns | Purpose |
|---|---|---|
| Identity/file | id, path, folder | stable short ID and relative path |
| Tags | title, artist, album, tag_genre | metadata read by Mutagen |
| File state | duration, size, mtime, status, error | scan signature and status |
| Raw analysis | raw_features vector(81) | direct audio-feature output |
| Language | lang_emb vector(256), lang_top JSONB | model output |
| Recommendation | embedding vector(81) | normalized similarity vector |
| Labels | lang_group, lang_name, lang_conf, mood, mood_cluster, arousal | build labels |
| Timestamps | created_at, updated_at | database bookkeeping |

status is intended to be new, ok, or failed:

    scan changed/new file -> new
    successful extraction -> ok
    feature extraction exception -> failed

### plays and model_state

plays stores an event when the browser starts a song and another completed event
when the song ends. session groups events from one browser session. The
recommender uses songs heard in the same session as a small co-play signal.

model_state stores the robust-scaling median, IQR, block weights, and song count
under key space. The current query uses the already stored embedding; this state
is useful for auditability and a future incremental implementation.

### What vector(81) means

A row might conceptually contain:

    raw_features = [mfcc_1_mean, ..., zero_crossing_rate]  # 81 numbers
    embedding    = [0.04, -0.11, ..., 0.02]                 # 81 numbers
    lang_emb     = [0.01, ..., -0.07]                       # 256 numbers

These are not human-readable descriptions. They are numerical coordinates used
to compare songs.

### Cosine distance and HNSW

The query in recommender.py uses:

    embedding <=> query_vector

With pgvector's vector_cosine_ops, <=> is cosine distance: lower is more
similar. The code converts it to a similarity-like number with 1 - distance.
Because build() L2-normalizes every stored embedding, dot product and cosine
similarity have the same ordering.

The HNSW index is a graph-based approximate-nearest-neighbor index. It avoids
comparing a query against every vector for a large library. The SQL still looks
like ORDER BY embedding <=> ... LIMIT 200; pgvector can use the index. For a
small library PostgreSQL may choose an exact scan, which is also correct.

### Connection pooling

Opening a database connection has setup cost. ConnectionPool keeps up to eight
reusable connections (min_size=1, max_size=8). An API request borrows one with
pool.connection(), runs its query, and returns it when the with block ends. The
pool is not a cache of query results and not a second database; it is a
controlled collection of database connections.

CLI code mostly uses one direct connection from db.connect(). Each connection
is passed to register_vector(con), which teaches psycopg how to convert between
PostgreSQL vectors and Python/NumPy values.

db.init() retries an unavailable database up to 60 times, one second apart. It
takes advisory lock 424242 so two processes do not create the schema
simultaneously. It creates the vector and pg_trgm extensions, tables, and
indexes, then returns None. If it never connects, it exits with
database not reachable.

arr(v) normalizes driver-specific vector results: None remains None, a pgvector
object with to_numpy() becomes a NumPy array, and other values go through
np.asarray(v).

## Stage 1: scanning files — pipeline.py, scan()

scan() is filesystem synchronization, not audio analysis.

1. MUSIC_DIR.rglob("*") recursively finds files.
2. The suffix is lowercased and checked against AUDIO_EXT.
3. p.relative_to(MUSIC_DIR).as_posix() produces a portable relative path, such
   as "Hindi/song.mp3".
4. p.stat() returns file size and modification time.
5. If the stored (size, mtime) is unchanged, the row is skipped.
6. _tags(p) reads title, artist, album, and genre through Mutagen. It returns a
   four-string tuple. Broken or unsupported tags return four empty strings.
7. probe_duration(p) returns seconds as a float.
8. song_id(rel) computes SHA-1 of the relative path and keeps 12 hex
   characters. For a fixed relative path, it is stable across runs.
9. INSERT ... ON CONFLICT(path) DO UPDATE creates or refreshes the row.
10. A changed file clears analysis vectors and becomes status new, forcing a
    later extract.
11. Files no longer present are deleted from songs; the foreign-key cascade
    also removes their plays rows.

Typical output:

    scan: 137 files, 12 new/changed, 2 removed

The 12 is not the total number of files; it is the number inserted or reset
because they were new or changed.

### Example row after scanning

For MUSIC_DIR/Hindi/Aaja.mp3, before extraction:

    id       = sha1("Hindi/Aaja.mp3")[:12]  # e.g. "8f1c..."
    path     = "Hindi/Aaja.mp3"
    folder   = "Hindi"
    title    = Mutagen title, or "Aaja" if no title tag exists
    duration = 214.73                         # example only
    status   = "new"
    raw_features/lang_emb/embedding = NULL

The exact ID, duration, and tags depend on the file.

## Stage 2: decoding audio — musicrec/audio.py

### probe_duration(path)

Returns one Python float in seconds. It first asks Mutagen for container
metadata. If that fails or returns no length, it runs ffprobe and parses its
output. If both fail, it returns 0.0 rather than raising.

### _run(cmd)

Returns:

    (samples, error_text)

samples is a one-dimensional float32 NumPy array created from ffmpeg's raw
little-endian 32-bit PCM stdout. error_text is stderr truncated to 200
characters.

### decode(path, start, dur, sr)

The ffmpeg options mean:

- -vn: ignore video streams;
- -ac 1: convert to mono;
- -ar sr: resample to the requested rate;
- -f f32le pipe:1: write raw float32 samples to stdout;
- -ss and -t: seek to start and decode dur seconds.

It tries fast input seeking, accurate output seeking, and finally decoding from
the beginning. The first non-empty result is returned. For 45 seconds at
22,050 Hz, the approximate output length is:

    45 * 22,050 = 992,250 samples

The actual count can differ slightly at file boundaries. If all attempts
produce no samples, it raises RuntimeError("ffmpeg produced no audio: ...").

## Stage 3: extracting audio features — musicrec/features.py

extract_features(path, duration) returns one np.float32 vector with shape
(81,). It selects a 45-second clip from the middle:

    off = max(0.0, duration / 2 - C.CLIP_SECONDS / 2)
    y = decode(path, off, C.CLIP_SECONDS, C.SR)

If the decoded clip has fewer than three seconds of samples, it raises
ValueError("audio shorter than 3 s"). The worker catches this and records a
failed song.

block_slices() returns:

    {
        "timbre": slice(0, 47),
        "harmony": slice(47, 71),
        "rhythm": slice(71, 75),
        "energy": slice(75, 81),
    }

The exact scalar values depend on the recording. Shapes and meanings are
stable:

| Block | Calculation | Size |
|---|---|---:|
| timbre | 20 MFCC means + 20 MFCC standard deviations + 7 spectral-contrast means | 47 |
| harmony | 12 chroma means + 12 chroma standard deviations | 24 |
| rhythm | tempo, onset mean, onset standard deviation, periodicity | 4 |
| energy | RMS mean/std, centroid, bandwidth, rolloff, zero-crossing rate | 6 |
| total | concatenated and converted to float32 | 81 |

Important intermediate outputs:

- librosa.effects.hpss(y) returns y_h (harmonic) and y_p (percussive), both
  one-dimensional arrays.
- mfcc has shape (20, frames); mean(1) and std(1) each produce 20 numbers.
- contrast normally has 7 rows; its mean produces 7 numbers.
- chroma has shape (12, frames); its mean and standard deviation produce 12
  numbers each.
- onset is a one-dimensional strength curve. _tempo returns one float,
  generally interpreted as BPM. Autocorrelation estimates periodicity; that
  result is one float.
- RMS, centroid, bandwidth, rolloff, and zero-crossing rate describe loudness
  or spectral shape. The code averages frame values to get one number each.
- np.nan_to_num(v) replaces NaN or infinite values with finite numbers.

The comments explain that a beat tracker was avoided because of heavy
numba/runtime problems; rhythm uses tempogram/autocorrelation instead.

## Stage 4: language identification — musicrec/langid.py

This stage is optional. C.LANGID_ENABLED is false when LANGID=0.

LangID.__init__() imports PyTorch and SpeechBrain, limits Torch to one CPU
thread, and loads the VoxLingua107 ECAPA model into
MODEL_DIR/voxlingua107. The first run may download model files. self.labels
maps model indexes to strings such as "hi: Hindi".

analyse(path, duration) chooses at most three eight-second windows:

- for a song of at most nine seconds, one starts at 0.0;
- otherwise windows are centered near 20%, 50%, and 75% of the song;
- each start is clamped so it stays within the file.

Each window is decoded at 16,000 Hz, so a full window contains about
8 * 16,000 = 128,000 samples. Windows shorter than two seconds or with RMS
below 1e-4 are skipped as too short/silent. Short surviving windows are padded
to exactly the model length.

If all windows are rejected, the function returns None.

Otherwise, it produces:

    emb : NumPy float32 array, shape (256,), approximately unit length
    top : list of up to five dictionaries
          {"code": "hi", "name": "Hindi", "p": 0.8123}

emb is the average of window embeddings and is L2-normalized. Model
log-probabilities are exponentiated into probabilities, averaged across
windows, normalized again, and sorted descending. top[0] is the highest
average model guess, not a guaranteed truth.

In pipeline._analyse, a model-load exception does not fail every song. It
prints a warning, sets the worker's language model to False, and continues
with ordinary audio features. In that case lang_emb and lang_top are null.

## Stage 5: parallel extraction — pipeline.py, extract()

_analyse(song_id, relative_path) is the worker function. Success returns:

    (song_id, features_81, language_result_or_None, None)

Failure returns:

    (song_id, None, None, "ExceptionType: message")

extract(retry=False, rebuild=False) selects songs whose raw_features IS NULL.
Unless retry=True, it excludes rows already marked failed. rebuild=True clears
all analysis vectors and resets every song to new before selecting work.

Work is sent to ProcessPoolExecutor using spawn. Processes are used because
audio decoding and numerical/model work are CPU-heavy. Each worker lazily
creates its own LangID model the first time it needs one. The first extraction
can therefore be slow and use substantial memory.

For each completed future:

- error -> status failed and the first 500 characters go into error;
- success -> raw_features, optional lang_emb, optional JSON lang_top, and
  status ok are written;
- every 25 completions, the connection commits; it also commits at the end of
  a pool round.

Typical output:

    extract: 137 songs, 4 workers (first run downloads the language model)
    features: 100%|##########| 137/137

If a process pool crashes, remaining work is retried for up to four rounds. A
song-level exception is not a pool crash; it becomes a failed row.

## Stage 6: building the shared recommendation space — pipeline.py, build()

build() reads all songs with raw_features IS NOT NULL. It requires at least
five extracted songs. With n songs:

    X shape = (n, 81)

### Robust normalization

For each of the 81 feature columns it calculates:

- med: the median;
- iqr: the 75th percentile minus the 25th percentile, plus a small epsilon;
- Z: (X - med) / (iqr / 1.349), clipped to [-4, 4].

This is a robust z-score. Median/IQR are less affected by unusual recordings
than mean/standard deviation.

### Block equalization and final embeddings

The blocks have different sizes, so each block is scaled by its configured
weight and divided by the square root of its size:

    W[:, block] = Z[:, block] * block_weight / sqrt(block_size)

This prevents the 47-dimensional timbre block from dominating the
4-dimensional rhythm block only because it has more values. The code then
centers every feature column and L2-normalizes every song row:

    E shape = (n, 81)
    norm(E[row]) approximately 1

E[row] is stored as songs.embedding and used by pgvector. raw_features is
retained so normalization can be rebuilt later.

arousal is a simple average of four normalized coordinates: tempo, onset
strength, RMS, and spectral centroid. It is a heuristic used to order mood
clusters, not a separately trained emotion model.

### Language groups

group_languages(LE, tops, folders) returns:

    (gid, language_names, confidence)

If fewer than max(10, 2 * LANG_MIN_GROUP) songs have a language embedding, the
code prints a warning and groups by folder_stem instead. With the default
LANG_MIN_GROUP=8, that threshold is 16 songs with language data.

With enough data:

1. songs with language embeddings are stacked into Xl with shape
   (songs_with_language, 256);
2. those vectors are normalized;
3. each lang_top list's top language is counted;
4. automatic k is based on frequent language names and clipped to 2 through
   10, unless LANG_K is provided;
5. KMeans assigns a cluster to each language vector;
6. each cluster receives its majority top-1 language name;
7. duplicate names receive a folder-stem or group-N disambiguating hint;
8. songs without embeddings use their folder's common cluster or become
   unknown.

KMeans cluster numbers are arbitrary implementation labels. lang_name is the
human-facing majority label. lang_conf is the probability of that label from
the song's top-five model output, or 0.0 when unavailable.

### Mood clusters

For each language group:

    k = clip(number_of_songs_in_group // 12, 1, 5)

Groups with fewer than 24 songs get k=1 and remain "mixed". Larger groups are
clustered using E. Clusters are sorted by average arousal and mapped to:

    calm / soft
    mellow / emotional
    groovy / mid-tempo
    upbeat
    energetic / party

These are relative heuristics, not ground-truth mood annotations.

Finally, build() updates every song and writes normalization metadata to
model_state. Example output:

    build: 137 songs -> 4 language groups: ['English', 'Hindi', 'Punjabi', 'unknown']

## Stage 7: reporting and evaluation

report() prints:

1. language group, song count, and mean language confidence;
2. folder x detected language, exposing mixed folders;
3. language x mood;
4. up to 30 failed files and stored error text.

evaluate(sample=200) selects up to 200 songs with embeddings, deterministically
using NumPy seed 0, and asks for 10 recommendations for each. Example:

    language-consistency@10 = 0.842   mood-consistency@10 = 0.517

These are internal consistency measurements, not measures of whether a human
likes the recommendations. A high score can happen because the system strongly
favors the same group.

## Recommendation logic — recommender.py

### API-shaped song output

_song(row) converts a SQL tuple into a JSON-friendly dictionary:

    {
        "id": "8f1c...",
        "title": "Aaja",
        "artist": "Some Artist",
        "folder": "Hindi",
        "duration": 214.73,
        "language": "Hindi",          # or "unknown"
        "lang_conf": 0.812,
        "mood": "upbeat",              # or "mixed"
        "lang_guess": ["Hindi 0.81", "Urdu 0.10", "..."],
    }

### search(q, limit, language)

The searchable text is lowercase title + artist + folder. If q is present,
SQL accepts either a substring match (LIKE) or PostgreSQL trigram
word_similarity(...) > 0.35, then sorts by similarity and title. The pg_trgm
extension created in db.py supplies that fuzzy-matching function.

It returns a Python list of _song dictionaries, not raw SQL rows.

### languages()

It returns a nested dictionary such as:

    {
      "Hindi": {"count": 80, "moods": {"upbeat": 40, "mixed": 40}},
      "English": {"count": 57, "moods": {"calm / soft": 20, "upbeat": 37}}
    }

The browser uses it to populate the language dropdown.

### recommend(sid, k, strict_language, seeds)

The algorithm has four parts:

1. Read the seed's lang_group, mood_cluster, and embedding. If the song does
   not exist or has no embedding, return [].
2. Start with the seed vector. For up to three recent IDs, add vectors with
   weights 0.5, 0.25, and 0.125, then normalize. This is session-aware while
   keeping the current song strongest.
3. Query up to 200 nearest candidates by cosine distance. With strict language,
   only the same lang_group is searched. Without it, same-group candidates are
   collected first and then general candidates; a dictionary removes duplicates.
4. Rerank:

    score = cosine_similarity
            + 0.25 if same language group
            + 0.05 if same mood cluster
            + co-play bonus when history supports it

The co-play bonus is weighted by 0.15 and uses logarithmic normalization.
Results receive rounded score and similarity fields, are sorted descending, and
the return value contains at most k dictionaries.

The score can be greater than 1.0 because it includes bonuses; it is not a
probability.

## HTTP API — api.py

| Route | Input | Output/behavior |
|---|---|---|
| GET /api/search | q, language, limit | list of song dictionaries |
| GET /api/similar/{sid} | k, strict_language, comma-separated seeds | recommendation list; 404 if missing |
| GET /api/languages | none | language -> count/mood dictionary |
| POST /api/play | JSON {session, song, completed} | {"ok": true} after insertion |
| GET /api/stream/{sid} | song ID | audio file response |
| GET / and static paths | browser request | static/index.html and assets |

FastAPI uses the Play Pydantic model to parse and validate the JSON body.
completed defaults to false if omitted.

stream() looks up the relative path by ID, joins it to MUSIC_DIR, guesses a MIME
type such as audio/mpeg, and returns FileResponse. The browser never needs to
know the real filesystem path.

## Browser flow — static/index.html

The HTML contains layout and CSS, then a JavaScript client beginning around
line 1326. The important runtime flow is:

1. A session ID is loaded from localStorage or generated with
   crypto.randomUUID().
2. init() calls /api/languages, fills the language dropdown, and calls search().
3. search() calls /api/search, receives dictionaries, and renders song rows.
4. Clicking a row calls playSong(song).
5. playSong sets the audio element source to /api/stream/{song_id}, starts
   playback, posts a non-completed play event, and calls /api/similar/{song_id}.
6. The recommendation response is rendered as more clickable rows.
7. The audio ended event posts a completed play event and advances to the next
   queue item unless repeat is enabled.

The browser sends recent song IDs in reverse order. The backend uses those IDs
as the decaying session context described above.

esc(...) HTML-escapes values before inserting them into template strings. That
matters because song tags are file/user-controlled text. Audio playback is
handled by the browser's audio element; Python only supplies the file.

## Docker and deployment flow

docker-compose.yml starts two services:

- db: PostgreSQL 16 with pgvector. Data is persisted in pgdata and host port
  5433 maps to container port 5432.
- app: the Python image. It waits for the database health check, mounts the host
  music directory read-only at /music, stores model downloads in the models
  volume, and exposes port 8000.

Inside the Docker network, the app uses
postgresql://musicrec:musicrec@db:5432/musicrec. From the host, psql uses
localhost:5433.

Dockerfile installs ffmpeg and libsndfile, CPU-only Torch/Torchaudio, then the
Python requirements, and copies musicrec into /app. HF_HOME=/models keeps
Hugging Face model files in the mounted model volume.

update.sh rebuilds the image, then runs:

    scan -> extract --retry -> build -> report -> start API

## Libraries in plain language

| Library/tool | What this project uses it for |
|---|---|
| NumPy | arrays, statistics, normalization, vector math |
| SciPy | indirect scientific dependency for the audio/ML stack |
| librosa | MFCC, chroma, spectral features, onset/rhythm |
| Mutagen | audio tags and often duration metadata |
| ffmpeg/ffprobe | decoding many containers/codecs and duration |
| PyTorch | tensor operations and running the language model |
| SpeechBrain | VoxLingua107 ECAPA language-identification model |
| scikit-learn | KMeans clustering |
| psycopg | PostgreSQL driver for Python |
| psycopg_pool | reusable PostgreSQL connection pool |
| pgvector | vector type, conversion, distance operators, HNSW index |
| FastAPI | HTTP route definitions and request validation |
| Uvicorn | ASGI server for FastAPI |
| tqdm | progress bars and worker messages |
| pydantic | validates the /api/play request model |

The project uses both feature extraction and machine-learning inference:
librosa calculates hand-designed numerical audio features, while SpeechBrain
runs a pretrained neural model for language. KMeans is unsupervised clustering:
it groups vectors by numerical closeness, then the code assigns human-readable
names.

## Current code-level findings

These are observations from the current repository snapshot:

1. Recommender.search() currently initializes where as an empty list. When both
   q and language are empty, line 35 builds SQL equivalent to:

       SELECT ... FROM songs WHERE ORDER BY title LIMIT ...

   PostgreSQL rejects that syntax. The browser calls this empty search during
   init(), so the initial library can fail until search is corrected. The
   immediately previous git commit shows that the base condition
   embedding IS NOT NULL was removed in this refactor. This guide does not
   change it because its purpose is documentation.

2. search() no longer filters embedding IS NOT NULL for non-empty searches.
   Songs can therefore appear after scan but before build. They can be displayed,
   but recommendations for one without an embedding return an empty list.

3. build() requires five extracted songs, while language grouping has a separate
   minimum-data fallback. A small library may scan and extract but still be
   unable to build.

4. Language and mood labels are heuristic. A language label is a majority label
   on a KMeans cluster, and a mood label is based on relative arousal ordering.
   They are not ground-truth annotations.

5. There is no test directory in this repository. A safe manual verification
   order is scan, inspect report, run extract, inspect failed rows, run build,
   inspect report again, then call API routes one at a time.

## Small mental simulation

Imagine two songs:

    Hindi/A.mp3 -> raw_features shape (81,), language embedding (256,)
    Hindi/B.mp3 -> raw_features shape (81,), language embedding (256,)

After build():

    A.embedding and B.embedding are normalized 81-value vectors
    A.lang_name = "Hindi"; B.lang_name = "Hindi"
    A.mood = "upbeat"; B.mood = "mellow / emotional"

When A is playing, the recommender computes A's vector, finds nearby stored
vectors, gives B a language bonus because both belong to the Hindi group, gives
it a mood bonus only if the mood clusters match, and optionally adds a small
co-play bonus. The API serializes that result to JSON, and the browser displays
it as a clickable row. The original audio is streamed only after the browser
chooses a song.

The complete conceptual loop is:

    file -> metadata -> waveform -> 81 features
         -> optional language model -> database vectors
         -> build normalization/clusters -> nearest-neighbor candidates
         -> reranking -> JSON -> browser playback
