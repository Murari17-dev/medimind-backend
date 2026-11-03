
import sqlite3
from pathlib import Path

database_path = Path(__file__).resolve().parent / "medimind.db"

connection = sqlite3.connect(database_path)
cursor = connection.cursor()

print("\n========== DATABASE INSPECTION ==========")

print("\nTABLES:")

tables = cursor.execute(
    "SELECT name FROM sqlite_master WHERE type = ?",
    ("table",)
).fetchall()

for table in tables:
    print("-", table[0])

print("\nRECORD COUNTS:")

for table_name in ("prescriptions", "medications"):
    count = cursor.execute(
        f"SELECT COUNT(*) FROM {table_name}"
    ).fetchone()[0]

    print(f"{table_name}: {count}")

print("\nPRESCRIPTIONS TABLE SCHEMA:")

for column in cursor.execute(
    "PRAGMA table_info(prescriptions)"
).fetchall():
    print(column)

print("\nMEDICATIONS TABLE SCHEMA:")

for column in cursor.execute(
    "PRAGMA table_info(medications)"
).fetchall():
    print(column)

print("\n=========================================")

connection.close()
