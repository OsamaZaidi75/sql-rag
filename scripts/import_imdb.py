"""
Downloads the official IMDb public datasets and builds a SQLite database
with millions of rows — a real-world scale test for SQL RAG.

Datasets (https://datasets.imdbws.com, CC0, no signup):
  - title.basics.tsv.gz   (~10M rows: movies, series, episodes)
  - title.ratings.tsv.gz  (~1.5M rows: averageRating, numVotes)

Run:
  python scripts/import_imdb.py [--out data/imdb.db] [--limit 100000]

--limit imports only the first N title rows (handy for a quick smoke test).
Full import takes a few minutes and needs ~2-3 GB free disk.

Example questions to try afterwards:
  - "Top 10 highest rated drama movies from the 1990s with at least 50000 votes"
  - "Average runtime of sci-fi series by decade"
  - "Which year had the most movie releases?"
"""
import argparse
import gzip
import os
import sqlite3
import sys
import urllib.request

BASE_URL = "https://datasets.imdbws.com"
FILES = ["title.basics.tsv.gz", "title.ratings.tsv.gz"]

SCHEMA = """
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

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_titles_year ON titles(start_year)",
    "CREATE INDEX IF NOT EXISTS idx_titles_type ON titles(title_type)",
    "CREATE INDEX IF NOT EXISTS idx_ratings_votes ON ratings(num_votes)",
]


def _to_int(v):
    return None if v in (None, "", "\\N") else int(v)


def _to_float(v):
    return None if v in (None, "", "\\N") else float(v)


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


def import_all(out_db: str, limit: int = 0, batch: int = 5000):
    basics_path = download("title.basics.tsv.gz")
    ratings_path = download("title.ratings.tsv.gz")

    if os.path.exists(out_db):
        print(f"Removing existing {out_db}")
        os.remove(out_db)
    con = sqlite3.connect(out_db)
    con.executescript("PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;")
    con.executescript(SCHEMA)

    n = 0
    rows = []
    for r in stream_tsv_rows(basics_path):
        rows.append((
            r["tconst"], r["titleType"], r["primaryTitle"], r["originalTitle"],
            _to_int(r["isAdult"]), _to_int(r["startYear"]), _to_int(r["endYear"]),
            _to_int(r["runtimeMinutes"]), None if r["genres"] == "\\N" else r["genres"],
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

    m = 0
    rows = []
    wanted = None
    if limit:
        wanted = {r[0] for r in con.execute("SELECT tconst FROM titles")}
    for r in stream_tsv_rows(ratings_path):
        if wanted is not None and r["tconst"] not in wanted:
            continue
        rows.append((r["tconst"], _to_float(r["averageRating"]), _to_int(r["numVotes"])))
        m += 1
        if len(rows) >= batch:
            con.executemany("INSERT OR IGNORE INTO ratings VALUES (?,?,?)", rows)
            rows = []
    if rows:
        con.executemany("INSERT OR IGNORE INTO ratings VALUES (?,?,?)", rows)
    con.commit()
    print(f"Imported {m:,} ratings")

    print("Creating indexes ...")
    for idx in INDEXES:
        con.execute(idx)
    con.commit()
    t = con.execute("SELECT COUNT(*) FROM titles").fetchone()[0]
    rr = con.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    con.close()
    print(f"Done: {out_db} — {t:,} titles, {rr:,} ratings")


def main():
    ap = argparse.ArgumentParser(description="Build a multi-million-row IMDb SQLite DB")
    ap.add_argument("--out", default="data/imdb.db")
    ap.add_argument("--limit", type=int, default=0,
                    help="import only first N titles (0 = all)")
    args = ap.parse_args()
    import_all(args.out, limit=args.limit)


if __name__ == "__main__":
    sys.exit(main())
