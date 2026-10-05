"""
Downloads the official IMDb public datasets and builds a SQLite database
with tens of millions of rows — a real-world scale test for SQL RAG.

Datasets (https://datasets.imdbws.com, CC0, no signup). Core (always imported):
  - title.basics.tsv.gz   (~10M rows: movies, series, episodes)
  - title.ratings.tsv.gz  (~1.5M rows: averageRating, numVotes)

Optional extras via --extras (comma-separated, or "all"):
  - principals  title.principals.tsv.gz (~60M rows: cast & crew per title)
  - crew        title.crew.tsv.gz       (~10M rows: directors, writers)
  - episode     title.episode.tsv.gz    (~8M rows: series -> episode mapping)
  - people      name.basics.tsv.gz     (~13M rows: actors, directors, ...)
  - akas        title.akas.tsv.gz      (~35M rows: alternate titles/regions)

Run:
  python scripts/import_imdb.py [--out data/imdb.db] [--limit 100000]
  python scripts/import_imdb.py --extras principals,crew,people
  python scripts/import_imdb.py --extras all   # everything: 100M+ rows, needs tens of GB free

--limit imports only the first N title rows (handy for a quick smoke test).
Full core import takes a few minutes; extras take significantly longer
(principals alone is several GB compressed). Re-running overwrites the DB.

Example questions to try afterwards:
  - "Top 10 highest rated drama movies from the 1990s with at least 50000 votes"
  - "Which year had the most movie releases?"
  - "Actors who appeared in the most movies"            (needs principals)
  - "Directors with the most titles"                   (needs crew)
  - "Longest-running series by episode count"          (needs episode)
"""
import argparse
import gzip
import os
import sqlite3
import sys
import urllib.request

BASE_URL = "https://datasets.imdbws.com"

CORE_SCHEMA = """
CREATE TABLE IF NOT EXISTS titles (
    tconst TEXT PRIMARY KEY,
    title_type TEXT,
    primary_title TEXT,
    original_title TEXT,
    is_adult INTEGER,
    start_year INTEGER,
    end_year INTEGER,
    runtime_minutes INTEGER,
    genres TEXT
);
CREATE TABLE IF NOT EXISTS ratings (
    tconst TEXT PRIMARY KEY,
    average_rating REAL,
    num_votes INTEGER,
    FOREIGN KEY (tconst) REFERENCES titles(tconst)
);
"""

CORE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_titles_year ON titles(start_year)",
    "CREATE INDEX IF NOT EXISTS idx_titles_type ON titles(title_type)",
    "CREATE INDEX IF NOT EXISTS idx_ratings_votes ON ratings(num_votes)",
]


def _to_int(v):
    return None if v in (None, "", "\\N") else int(v)


def _to_float(v):
    return None if v in (None, "", "\\N") else float(v)


def _to_text(v):
    return None if v in (None, "", "\\N") else v


def download(path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"Reusing existing {path}")
        return path
    url = f"{BASE_URL}/{os.path.basename(path)}"
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, path)
    print(f"Saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def stream_tsv_rows(gz_path: str):
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            yield dict(zip(header, line.rstrip("\n").split("\t")))


def bulk_load(con, gz_path, table, columns, mapper, batch=5000, skip=None):
    """Stream a .tsv.gz into a table. skip(tconst_set, row) -> True drops the row."""
    placeholders = ",".join("?" * len(columns))
    sql = f"INSERT OR IGNORE INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    n = 0
    rows = []
    for r in stream_tsv_rows(gz_path):
        if skip is not None and skip(r):
            continue
        rows.append(mapper(r))
        n += 1
        if len(rows) >= batch:
            con.executemany(sql, rows)
            rows = []
            print(f"  {table}: {n:,}", end="\r")
    if rows:
        con.executemany(sql, rows)
    con.commit()
    print(f"\nImported {n:,} rows into {table}")
    return n


# ---------------------------------------------------------------- extras ---
EXTRAS = {
    "principals": {
        "file": "title.principals.tsv.gz",
        "table": "principals",
        "columns": ["tconst", "ordering", "nconst", "category", "job", "characters"],
        "schema": """CREATE TABLE IF NOT EXISTS principals (
            tconst TEXT, ordering INTEGER, nconst TEXT,
            category TEXT, job TEXT, characters TEXT,
            PRIMARY KEY (tconst, ordering))""",
        "indexes": [
            "CREATE INDEX IF NOT EXISTS idx_principals_nconst ON principals(nconst)",
            "CREATE INDEX IF NOT EXISTS idx_principals_category ON principals(category)",
            "CREATE INDEX IF NOT EXISTS idx_principals_cat_nconst ON principals(category, nconst)",
        ],
        "mapper": lambda r: (r["tconst"], _to_int(r["ordering"]), r["nconst"],
                             r["category"], _to_text(r["job"]), _to_text(r["characters"])),
        "desc": "~60M rows: cast & crew per title",
    },
    "crew": {
        "file": "title.crew.tsv.gz",
        "table": "crew",
        "columns": ["tconst", "directors", "writers"],
        "schema": """CREATE TABLE IF NOT EXISTS crew (
            tconst TEXT PRIMARY KEY, directors TEXT, writers TEXT)""",
        "indexes": [],
        "mapper": lambda r: (r["tconst"], _to_text(r["directors"]), _to_text(r["writers"])),
        "desc": "~10M rows: directors & writers per title",
    },
    "episode": {
        "file": "title.episode.tsv.gz",
        "table": "episodes",
        "columns": ["tconst", "parent_tconst", "season_number", "episode_number"],
        "schema": """CREATE TABLE IF NOT EXISTS episodes (
            tconst TEXT PRIMARY KEY, parent_tconst TEXT,
            season_number INTEGER, episode_number INTEGER)""",
        "indexes": ["CREATE INDEX IF NOT EXISTS idx_episodes_parent ON episodes(parent_tconst)"],
        "mapper": lambda r: (r["tconst"], r["parentTconst"],
                             _to_int(r["seasonNumber"]), _to_int(r["episodeNumber"])),
        "desc": "~8M rows: episode -> series mapping",
    },
    "people": {
        "file": "name.basics.tsv.gz",
        "table": "people",
        "columns": ["nconst", "primary_name", "birth_year", "death_year", "primary_profession"],
        "schema": """CREATE TABLE IF NOT EXISTS people (
            nconst TEXT PRIMARY KEY, primary_name TEXT,
            birth_year INTEGER, death_year INTEGER, primary_profession TEXT)""",
        "indexes": ["CREATE INDEX IF NOT EXISTS idx_people_name ON people(primary_name)"],
        "mapper": lambda r: (r["nconst"], r["primaryName"],
                             _to_int(r["birthYear"]), _to_int(r["deathYear"]),
                             _to_text(r["primaryProfession"])),
        "desc": "~13M rows: people (actors, directors, ...)",
    },
    "akas": {
        "file": "title.akas.tsv.gz",
        "table": "akas",
        "columns": ["title_id", "ordering", "title", "region", "language", "types", "is_original_title"],
        "schema": """CREATE TABLE IF NOT EXISTS akas (
            title_id TEXT, ordering INTEGER, title TEXT, region TEXT,
            language TEXT, types TEXT, is_original_title INTEGER,
            PRIMARY KEY (title_id, ordering))""",
        "indexes": ["CREATE INDEX IF NOT EXISTS idx_akas_region ON akas(region)"],
        "mapper": lambda r: (r["titleId"], _to_int(r["ordering"]), r["title"],
                             _to_text(r["region"]), _to_text(r["language"]),
                             _to_text(r["types"]), _to_int(r["isOriginalTitle"])),
        "desc": "~35M rows: alternate titles by region/language",
    },
}


def import_all(out_db: str, limit: int = 0, extras=(), batch: int = 5000):
    if os.path.exists(out_db):
        print(f"Removing existing {out_db}")
        os.remove(out_db)
    con = sqlite3.connect(out_db)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    con.executescript(CORE_SCHEMA)

    # ---- core: titles ----
    basics_path = download("title.basics.tsv.gz")
    n = 0
    rows = []
    for r in stream_tsv_rows(basics_path):
        rows.append((
            r["tconst"], r["titleType"], r["primaryTitle"], r["originalTitle"],
            _to_int(r["isAdult"]), _to_int(r["startYear"]), _to_int(r["endYear"]),
            _to_int(r["runtimeMinutes"]), _to_text(r["genres"]),
        ))
        n += 1
        if len(rows) >= batch:
            con.executemany(
                "INSERT OR IGNORE INTO titles VALUES (?,?,?,?,?,?,?,?,?)", rows)
            rows = []
            print(f"  titles: {n:,}", end="\r")
        if limit and n >= limit:
            break
    if rows:
        con.executemany("INSERT OR IGNORE INTO titles VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    print(f"\nImported {n:,} titles")

    # ---- core: ratings ----
    ratings_path = download("title.ratings.tsv.gz")
    wanted = None
    if limit:
        wanted = {r[0] for r in con.execute("SELECT tconst FROM titles")}
    bulk_load(con, ratings_path, "ratings",
              ["tconst", "average_rating", "num_votes"],
              lambda r: (r["tconst"], _to_float(r["averageRating"]), _to_int(r["numVotes"])),
              batch=batch,
              skip=(lambda r: wanted is not None and r["tconst"] not in wanted))

    # ---- extras ----
    for name in extras:
        spec = EXTRAS[name]
        print(f"\n--- extra: {name} ({spec['desc']}) ---")
        con.executescript(spec["schema"])
        path = download(spec["file"])
        bulk_load(con, path, spec["table"], spec["columns"], spec["mapper"], batch=batch)
        for idx in spec["indexes"]:
            print(f"  index: {idx.split('ON')[1].strip()}")
            con.execute(idx)
        con.commit()

    print("\nCreating core indexes ...")
    for idx in CORE_INDEXES:
        con.execute(idx)
    con.commit()

    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    total = 0
    print(f"\nDone: {out_db}")
    for t in tables:
        c = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        total += c
        print(f"  {t}: {c:,} rows")
    print(f"  TOTAL: {total:,} rows")
    con.close()


def main():
    ap = argparse.ArgumentParser(description="Build a multi-million-row IMDb SQLite DB")
    ap.add_argument("--out", default="data/imdb.db")
    ap.add_argument("--limit", type=int, default=0,
                    help="import only first N titles (0 = all)")
    ap.add_argument("--extras", default="",
                    help=f"comma-separated extras ({', '.join(EXTRAS)}) or 'all'")
    args = ap.parse_args()
    extras = []
    if args.extras.strip().lower() == "all":
        extras = list(EXTRAS)
    elif args.extras.strip():
        extras = [e.strip().lower() for e in args.extras.split(",")]
        unknown = [e for e in extras if e not in EXTRAS]
        if unknown:
            sys.exit(f"unknown extras: {unknown} (choose from {', '.join(EXTRAS)})")
    import_all(args.out, limit=args.limit, extras=extras)


if __name__ == "__main__":
    sys.exit(main())
