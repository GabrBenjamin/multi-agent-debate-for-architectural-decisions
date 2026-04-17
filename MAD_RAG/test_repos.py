#!/usr/bin/env python3
"""
Simple script to test repository accessibility before running the full extraction.
This will help identify which repositories are accessible and which are not.
"""

import os
import subprocess
import pandas as pd
from database import Database

# Set OpenAI API key for this process
os.environ["OPENAI_API_KEY"] = "REDACTED_API_KEY"

def test_repository_access(github_address, timeout=30):
    """Test if a repository is accessible"""
    try:
        result = subprocess.run(
            ['git', 'ls-remote', '--heads', github_address], 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        return result.returncode == 0, result.stderr if result.returncode != 0 else "OK"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)

def main():
    # Connect to the database
    db_path = os.path.join(os.getcwd(), 'main_dataset.db')
    db = Database(db_path)
    
    # Get the progress table
    df = db.to_df('progress')
    
    if df.empty:
        print("No data found in progress table")
        return
    
    # Filter for repositories that need extraction
    df = df[df['extraction_status'] != 'success']
    df = df[df['first_commit_hash'] != 'no_commit_hash']
    
    print(f"Testing {len(df)} repositories for accessibility...")
    print("=" * 60)
    
    accessible_count = 0
    inaccessible_count = 0
    
    for idx, row in df.head(10).iterrows():  # Test first 10 repositories
        github_address = row['github_address']
        repo_name = row['repo_name']
        
        print(f"Testing {repo_name} ({github_address})...", end=" ")
        
        is_accessible, error_msg = test_repository_access(github_address)
        
        if is_accessible:
            print("✓ ACCESSIBLE")
            accessible_count += 1
        else:
            print(f"✗ NOT ACCESSIBLE: {error_msg}")
            inaccessible_count += 1
    
    print("=" * 60)
    print(f"Summary:")
    print(f"  Accessible: {accessible_count}")
    print(f"  Not accessible: {inaccessible_count}")
    print(f"  Total tested: {accessible_count + inaccessible_count}")

if __name__ == "__main__":
    main()
