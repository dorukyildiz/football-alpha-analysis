import boto3
import os
from datetime import datetime

S3_BUCKET = 'football-alpha-analysis-doruk'
S3_KEY = 'data/players_data.csv'
LOCAL_FILE = 'data/players_data.csv'


def upload_to_s3():
    """Upload CSV to S3"""

    if not os.path.exists(LOCAL_FILE):
        print(f"[ERROR] File not found: {LOCAL_FILE}")
        return False

    s3 = boto3.client('s3')

    try:
        print(f"Uploading {LOCAL_FILE} to s3://{S3_BUCKET}/{S3_KEY}")
        s3.upload_file(LOCAL_FILE, S3_BUCKET, S3_KEY)
        print(f"[OK] Upload complete!")
        print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return False


if __name__ == "__main__":
    upload_to_s3()