"""
==================================================================================
 02_sql_analysis.py — Execute SQL Queries on Telecom Churn Dataset
==================================================================================
 Project  : Customer Churn Analysis for Telecom Industry
 Intern   : Data Analyst Internship — Elevate Labs
 Purpose  : Loads the CSV into an in-memory SQLite database and runs all 10
            analytical queries from sql/churn_queries.sql. Results are saved
            as individual sheets in an Excel workbook.

 Why SQLite?
   In real-world telecom analytics, data lives in SQL databases (Oracle, MySQL,
   Snowflake). Using SQLite here demonstrates the same SQL skills while keeping
   the project self-contained — no external DB setup required.
==================================================================================
"""

import os
import sqlite3
import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(__file__)
DATA_PATH   = os.path.join(BASE_DIR, "data", "telecom_churn.csv")
SQL_PATH    = os.path.join(BASE_DIR, "sql", "churn_queries.sql")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "sql_aggregations.xlsx")


def load_and_execute():
    """Load dataset into SQLite, execute all queries, and export results."""

    print("=" * 60)
    print(" 🗃️  Running SQL Aggregations on Churn Dataset")
    print("=" * 60)

    # ─── Load CSV ────────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    print(f"  ✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # ─── Create In-Memory SQLite DB ──────────────────────────────────────
    conn = sqlite3.connect(":memory:")
    df.to_sql("telecom_churn", conn, index=False, if_exists="replace")
    print("  ✅ Loaded data into SQLite table 'telecom_churn'")

    # ─── Read SQL File ───────────────────────────────────────────────────
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_content = f.read()

    # ─── Parse Individual Queries ────────────────────────────────────────
    # Split by the query header comments to extract individual queries
    queries = []
    current_query = []
    query_names = []

    for line in sql_content.split("\n"):
        stripped = line.strip()
        # Detect query headers (-- QUERY N: ...)
        if stripped.startswith("-- QUERY") and ":" in stripped:
            if current_query:
                # Save previous query
                query_sql = "\n".join(current_query).strip()
                if query_sql:
                    queries.append(query_sql)
            current_query = []
            # Extract query name
            name = stripped.split(":", 1)[1].strip()
            query_names.append(name)
        elif stripped.startswith("SELECT"):
            current_query.append(line)
        elif current_query:  # Continue adding lines to current query
            current_query.append(line)

    # Don't forget the last query
    if current_query:
        query_sql = "\n".join(current_query).strip()
        if query_sql:
            queries.append(query_sql)

    print(f"  ✅ Parsed {len(queries)} SQL queries from {SQL_PATH}")

    # ─── Execute Queries & Export ────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use openpyxl engine for xlsx format
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for i, (name, query) in enumerate(zip(query_names, queries), 1):
            try:
                result_df = pd.read_sql_query(query, conn)
                # Sheet name max 31 chars in Excel
                sheet_name = f"Q{i}_{name[:25]}"
                sheet_name = sheet_name.replace("/", "-").replace("\\", "-")
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)

                print(f"\n  📋 Query {i}: {name}")
                print(f"     Rows returned: {len(result_df)}")
                print(result_df.to_string(index=False))
            except Exception as e:
                print(f"  ❌ Query {i} failed: {e}")

    print(f"\n  ✅ All results saved to: {OUTPUT_XLSX}")

    conn.close()
    print("=" * 60)


# ─── Entry Point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_and_execute()
