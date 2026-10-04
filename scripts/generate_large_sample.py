"""
Generates a larger synthetic enterprise database for scale/performance demos.

Creates data/large_sample.db with the same 4-table schema as the sample DB but
~17k rows (12 departments, 1,500 employees, 150 projects, 15,000 sales
transactions across 2022-2024), so charts, aggregations, and query latency
feel real. Deterministic (seeded) so results are reproducible.

Run:  python scripts/generate_large_sample.py [--db data/large_sample.db]
"""
import argparse
import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from faker import Faker
except ImportError:
    sys.exit("faker is required: pip install faker")

SEED = 42
DEFAULT_DB = "data/large_sample.db"

DEPARTMENTS = [
    ("Engineering", 1_200_000.0, "San Francisco"),
    ("Marketing", 450_000.0, "New York"),
    ("Sales", 850_000.0, "Chicago"),
    ("Human Resources", 250_000.0, "San Francisco"),
    ("Product & Design", 600_000.0, "Austin"),
    ("Finance", 350_000.0, "New York"),
    ("Customer Success", 300_000.0, "Denver"),
    ("Legal", 280_000.0, "Washington"),
    ("Data Science", 700_000.0, "Seattle"),
    ("Operations", 400_000.0, "Chicago"),
    ("Security", 520_000.0, "Austin"),
    ("Research", 900_000.0, "Boston"),
]

ROLES = [
    "Software Engineer", "Senior Software Engineer", "Engineering Manager",
    "Data Analyst", "Product Manager", "Designer", "Account Executive",
    "Sales Manager", "Marketing Specialist", "HR Partner", "Financial Analyst",
    "DevOps Engineer", "QA Engineer", "Support Specialist",
]

CATEGORIES = ["Enterprise SaaS", "Cloud Storage", "Security Suite", "AI Analytics", "Consulting"]
REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]
STATUSES = ["Completed", "In Progress", "Planning", "On Hold"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--employees", type=int, default=1500)
    parser.add_argument("--projects", type=int, default=150)
    parser.add_argument("--sales", type=int, default=15000)
    args = parser.parse_args()

    fake = Faker()
    Faker.seed(SEED)
    random.seed(SEED)

    os.makedirs(os.path.dirname(args.db) or ".", exist_ok=True)
    if os.path.exists(args.db):
        os.remove(args.db)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()

    cur.execute("""CREATE TABLE departments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL UNIQUE,
        budget REAL NOT NULL, location TEXT NOT NULL, created_at TEXT NOT NULL)""")
    cur.execute("""CREATE TABLE employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        department_id INTEGER NOT NULL, role TEXT NOT NULL, salary REAL NOT NULL,
        hire_date TEXT NOT NULL, city TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
        performance_rating REAL NOT NULL,
        FOREIGN KEY (department_id) REFERENCES departments (id))""")
    cur.execute("""CREATE TABLE projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
        department_id INTEGER NOT NULL, lead_id INTEGER NOT NULL,
        budget REAL NOT NULL, status TEXT NOT NULL,
        start_date TEXT NOT NULL, end_date TEXT,
        FOREIGN KEY (department_id) REFERENCES departments (id),
        FOREIGN KEY (lead_id) REFERENCES employees (id))""")
    cur.execute("""CREATE TABLE sales_transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, employee_id INTEGER NOT NULL,
        customer_name TEXT NOT NULL, product_category TEXT NOT NULL,
        amount REAL NOT NULL, units_sold INTEGER NOT NULL,
        transaction_date TEXT NOT NULL, region TEXT NOT NULL,
        FOREIGN KEY (employee_id) REFERENCES employees (id))""")

    dept_rows = [(n, b, loc, f"20{random.randint(15, 20)}-{random.randint(1,12):02d}-01")
                  for n, b, loc in DEPARTMENTS]
    cur.executemany(
        "INSERT INTO departments (name, budget, location, created_at) VALUES (?,?,?,?)",
        dept_rows)
    dept_ids = list(range(1, len(dept_rows) + 1))
    dept_city = {i + 1: loc for i, (_, _, loc) in enumerate(DEPARTMENTS)}

    emp_rows = []
    for i in range(args.employees):
        d = random.choice(dept_ids)
        name = fake.name()
        emp_rows.append((
            name, d, random.choice(ROLES),
            round(random.uniform(55_000, 180_000), 2),
            fake.date_between(start_date="-6y", end_date="today").isoformat(),
            dept_city[d],
            f"{name.lower().replace(' ', '.').replace(chr(39), '')}{i}@enterprise.com",
            round(random.uniform(2.8, 5.0), 1),
        ))
    cur.executemany(
        """INSERT INTO employees (name, department_id, role, salary, hire_date,
           city, email, performance_rating) VALUES (?,?,?,?,?,?,?,?)""",
        emp_rows)

    proj_rows = []
    for i in range(args.projects):
        d = random.choice(dept_ids)
        start = fake.date_between(start_date="-3y", end_date="today")
        proj_rows.append((
            f"{fake.catch_phrase()} [{i+1}]", d, random.randint(1, args.employees),
            round(random.uniform(40_000, 400_000), 2),
            random.choices(STATUSES, weights=[45, 30, 15, 10])[0],
            start.isoformat(),
            (fake.date_between(start_date=start, end_date="+2y")).isoformat()
             if random.random() > 0.2 else None),
        )
    cur.executemany(
        """INSERT INTO projects (name, department_id, lead_id, budget, status,
           start_date, end_date) VALUES (?,?,?,?,?,?,?)""",
        proj_rows)

    sales_rows = []
    for _ in range(args.sales):
        sales_rows.append((
            random.randint(1, args.employees),
            fake.company(),
            random.choice(CATEGORIES),
            round(random.uniform(5_000, 150_000), 2),
            random.randint(1, 8),
            fake.date_between(start_date="-3y", end_date="today").isoformat(),
            random.choices(REGIONS, weights=[45, 25, 15, 10, 5])[0],
        ))
    cur.executemany(
        """INSERT INTO sales_transactions (employee_id, customer_name, product_category,
           amount, units_sold, transaction_date, region) VALUES (?,?,?,?,?,?,?)""",
        sales_rows)

    conn.commit()
    for t in ("departments", "employees", "projects", "sales_transactions"):
        n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t}: {n:,} rows")
    conn.close()
    size_mb = os.path.getsize(args.db) / 1e6
    print(f"Wrote {args.db} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
