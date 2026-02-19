#!/usr/bin/env python3
import sys
import argparse
from datetime import datetime


def run_full_pipeline(args):
    print(f"\nFOOTBALL ALPHA ANALYSIS PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not args.understat_only and not args.merge_only:
        print("STEP 1: Scraping FBref...")
        from fbref_scraper import run_scraper
        if run_scraper() is None:
            sys.exit(1)

    if not args.merge_only:
        print("\nSTEP 2: Scraping Understat...")
        from understat_scraper import run_understat_scraper
        if run_understat_scraper() is None:
            sys.exit(1)

    print("\nSTEP 3: Merging data...")
    from merge_data import run_merge
    if run_merge() is None:
        sys.exit(1)

    if not args.no_upload:
        print("\nSTEP 4: Uploading to S3...")
        from upload_to_s3 import upload_to_s3
        upload_to_s3()

    print(f"\nDONE! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--understat-only', action='store_true')
    parser.add_argument('--merge-only', action='store_true')
    parser.add_argument('--no-upload', action='store_true')
    run_full_pipeline(parser.parse_args())