import boto3
import os
from datetime import datetime

S3_BUCKET = 'football-alpha-analysis-doruk'

# File mapping: local path -> S3 key
UPLOADS = {
    'data/players_data.csv': 'merged/players_data.csv',
    'data/fbref_players.csv': 'raw/fbref_players.csv',
    'data/understat_players.csv': 'raw/understat_players.csv',
    'data/understat_raw.csv': 'raw/understat_raw.csv',
}


def upload_to_s3():
    """Upload all data files to S3"""

    s3 = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    success = 0
    failed = 0

    for local_path, s3_key in UPLOADS.items():
        if not os.path.exists(local_path):
            print(f"[SKIP] Not found: {local_path}")
            continue

        try:
            print(f"Uploading {local_path} -> s3://{S3_BUCKET}/{s3_key}")
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"  [OK]")
            success += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print(f"\nDone! {success} uploaded, {failed} failed ({timestamp})")
    return failed == 0


if __name__ == "__main__":
    upload_to_s3()