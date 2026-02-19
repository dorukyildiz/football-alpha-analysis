import boto3
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
S3_BUCKET = 'football-alpha-analysis-doruk'

FILES_TO_UPLOAD = {
    'players_data.csv': 'data/players_data.csv',
    'fbref_players.csv': 'data/fbref_players.csv',
    'understat_players.csv': 'data/understat_players.csv',
}


def upload_to_s3():
    s3 = boto3.client('s3')
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    for filename, s3_key in FILES_TO_UPLOAD.items():
        local_path = DATA_DIR / filename
        if not local_path.exists():
            print(f"[SKIP] Not found: {local_path}")
            continue
        try:
            s3.upload_file(str(local_path), S3_BUCKET, s3_key)
            print(f"[OK] {filename} -> s3://{S3_BUCKET}/{s3_key}")

            archive_key = f"archive/{filename.replace('.csv', '')}_{ts}.csv"
            s3.upload_file(str(local_path), S3_BUCKET, archive_key)
            print(f"[OK] Archived: {archive_key}")
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    upload_to_s3()