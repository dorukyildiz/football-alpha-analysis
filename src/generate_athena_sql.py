"""
Generate Athena CREATE TABLE SQL for players_data.csv
Reads the actual CSV header dynamically — no hardcoded column list.

Usage:
    python generate_athena_sql.py                          # reads from data/players_data.csv
    python generate_athena_sql.py path/to/players_data.csv # reads from custom path
"""

import re
import os
import sys
from pathlib import Path

# Where to look for the CSV
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PATHS = [
    PROJECT_ROOT / 'data' / 'players_data.csv',
    Path('data/players_data.csv'),
    Path('../data/players_data.csv'),
]

# Athena config
DATABASE = 'football_analytics'
TABLE = 'players'
S3_LOCATION = 's3://football-alpha-analysis-doruk/data/'

# Columns that should be STRING (not DOUBLE)
STRING_COLUMNS = {
    'player', 'nation', 'pos', 'squad', 'comp',
    'player_norm', 'squad_norm', 'player_clean', 'squad_clean',
}


def find_csv(custom_path=None):
    """Find players_data.csv"""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    for p in DEFAULT_PATHS:
        if os.path.exists(p):
            return str(p)
    return None


def read_header(csv_path):
    """Read first line of CSV to get column names"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
    columns = []
    for col in header_line.split(','):
        col = col.strip().strip('"')
        columns.append(col)
    return columns


def clean_column_name(name):
    """Convert CSV header to valid Athena column name"""
    name = name.lower().strip()
    name = name.replace('+', '_plus_')
    name = name.replace('-', '_')
    name = name.replace('/', '_per_')
    name = name.replace('%', 'pct')
    name = name.replace('#', 'num_')
    name = name.replace(' ', '_')
    name = re.sub(r'[^a-z0-9_]', '', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')

    if not name:
        return 'empty_col'
    if name[0].isdigit():
        name = 'col_' + name
    return name


def determine_type(clean_name):
    """Determine if column should be STRING or DOUBLE"""
    base_name = clean_name.split('_stats_')[0] if '_stats_' in clean_name else clean_name
    if base_name in STRING_COLUMNS or clean_name in STRING_COLUMNS:
        return 'string'
    return 'double'


def generate_sql(csv_path):
    """Generate CREATE TABLE SQL from CSV header"""
    raw_columns = read_header(csv_path)

    print(f"Read {len(raw_columns)} columns from: {csv_path}")

    used_names = {}
    col_definitions = []

    for raw_col in raw_columns:
        clean_name = clean_column_name(raw_col)

        if clean_name in used_names:
            used_names[clean_name] += 1
            clean_name = f"{clean_name}_{used_names[clean_name]}"
        else:
            used_names[clean_name] = 1

        dtype = determine_type(clean_name)
        col_definitions.append((clean_name, dtype, raw_col))

    col_lines = [f"  `{name}` {dtype}" for name, dtype, _ in col_definitions]

    sql = f"""-- Auto-generated Athena CREATE TABLE
-- Source: {os.path.basename(csv_path)}
-- Columns: {len(col_definitions)}
-- S3: {S3_LOCATION}

DROP TABLE IF EXISTS {DATABASE}.{TABLE};

CREATE EXTERNAL TABLE {DATABASE}.{TABLE} (
{(',{0}'.format(chr(10))).join(col_lines)}
)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES (
  'separatorChar' = ',',
  'quoteChar' = '"',
  'escapeChar' = '\\\\'
)
STORED AS TEXTFILE
LOCATION '{S3_LOCATION}'
TBLPROPERTIES (
  'skip.header.line.count'='1',
  'serialization.null.format'='',
  'use.null.for.invalid.data'='true'
);

-- Verification queries
SELECT COUNT(*) AS total_players FROM {DATABASE}.{TABLE};

SELECT player, squad, comp, gls, xg, finishing_alpha
FROM {DATABASE}.{TABLE}
WHERE CAST(gls AS DOUBLE) > 10
ORDER BY CAST(finishing_alpha AS DOUBLE) DESC
LIMIT 20;
"""

    return sql, col_definitions


def print_column_mapping(col_definitions):
    """Print column mapping for reference"""
    print(f"\n{'#':<4} {'Athena Column':<30} {'Type':<8} {'CSV Header':<30}")
    print("-" * 75)
    for i, (name, dtype, raw) in enumerate(col_definitions):
        marker = " *" if dtype == 'string' else ""
        print(f"{i:<4} {name:<30} {dtype:<8} {raw:<30}{marker}")
    print(f"\n* = string columns | Total: {len(col_definitions)} columns")


def main():
    custom_path = sys.argv[1] if len(sys.argv) > 1 else None
    csv_path = find_csv(custom_path)

    if csv_path is None:
        print("[ERROR] Could not find players_data.csv")
        print("Usage: python generate_athena_sql.py [path/to/players_data.csv]")
        print(f"Searched: {[str(p) for p in DEFAULT_PATHS]}")
        sys.exit(1)

    print(f"Reading CSV: {csv_path}")
    sql, col_definitions = generate_sql(csv_path)

    print_column_mapping(col_definitions)

    print("\n" + "=" * 60)
    print("GENERATED SQL:")
    print("=" * 60)
    print(sql)

    output_path = 'create_table.sql'
    with open(output_path, 'w') as f:
        f.write(sql)
    print(f"\n[OK] SQL saved to: {output_path} ({len(col_definitions)} columns)")


if __name__ == "__main__":
    main()