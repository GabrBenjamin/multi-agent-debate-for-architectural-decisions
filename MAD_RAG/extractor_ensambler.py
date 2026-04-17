import os
import json
import csv
import asyncio
import shutil
from tqdm import trange, tqdm
import sqlite3
import git
from git import Repo
import sys
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor, TimeoutError
from database import Database
from extractor import Extractor
# from view_data import ViewData
import pandas as pd
import traceback
import openpyxl
import time
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import create_engine, MetaData, Table, Column, String, Integer, Float, PrimaryKeyConstraint, case, delete
from sqlalchemy.sql import select, func, text
import pandas as pd
from debate_manager import *
from datetime import datetime

# Set OpenAI API key for this process
os.environ["OPENAI_API_KEY"] = "REDACTED_API_KEY"

sys.setrecursionlimit(2000)

class ExtractorEnsambler():
    def __init__(self):
        self.default_path = os.getcwd()
        self.db_path = os.path.join(self.default_path, 'main_dataset.db')
        self.db = Database(self.db_path)
        self.create_tables()
        
    def create_tables(self):
        create_metric_table = f"""
            CREATE TABLE IF NOT EXISTS progress (  
            repo_name TEXT,  
            github_address TEXT,  
            adr_name TEXT,
            commit_hash TEXT,
            first_commit_hash TEXT,
            commit_offset INTEGER,
            extraction_status TEXT,  
            context_considered_drivers TEXT,
            other_sections TEXT,
            supported_side TEXT,
            debate_answer TEXT,
            reason TEXT,
            message_history TEXT,
            timestamp DATETIME,
            used_head_fallback BOOLEAN,
            embedding_model TEXT,
            file_count INTEGER,
            status TEXT,
            PRIMARY KEY (repo_name, github_address, commit_hash, context_considered_drivers, other_sections)  
        );
        """
        self.db.execute_sql(create_metric_table)
        
        # Add missing columns to existing table if they don't exist
        try:
            self.db.execute_sql("ALTER TABLE progress ADD COLUMN used_head_fallback BOOLEAN")
        except:
            pass  # Column already exists
        try:
            self.db.execute_sql("ALTER TABLE progress ADD COLUMN embedding_model TEXT")
        except:
            pass  # Column already exists
        try:
            self.db.execute_sql("ALTER TABLE progress ADD COLUMN file_count INTEGER")
        except:
            pass  # Column already exists
        try:
            self.db.execute_sql("ALTER TABLE progress ADD COLUMN status TEXT")
        except:
            pass  # Column already exists
        
    def get_adr_info(self):
        self.grab_paths_from_source()
        self.get_adr_hash()

    def grab_paths_from_source(self, source=os.path.join(os.getcwd(), 'data_hash_only.csv')):
        df = self.db.to_df('progress')
        
        if 'commit_hash' in df.columns and len(df) > 0: 
            adr_ids = (df['commit_hash'] + '_adr_id_' + df['adr_name'] + '_url_' + df['github_address'] + '_context_' + df['context_considered_drivers']).tolist()
        else:
            adr_ids = []
            
        with open(source, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for i, row in enumerate(csv_reader):
                adr_id = row['hash'] + '_adr_id_' + row['path'] + '_url_' + row['repositoryUrl'] + '_context_' + row['context_considered_drivers']
                if adr_id in adr_ids:
                    continue
                
                repo_name = str(i)+row['repositoryUrl'].split('github.com/')[1].split('.git')[0].replace('/','_').replace('.','').replace('-','')
                value = {
                    'github_address': row['repositoryUrl'],
                    'repo_name': repo_name,
                    'adr_name': row['path'],
                    'commit_hash': row['hash'],
                    'first_commit_hash': None,
                    'commit_offset': None,
                    'extraction_status': 'false',
                    'context_considered_drivers':  row['context_considered_drivers'],
                    'other_sections':  row['other_sections'],
                    'supported_side': None,
                    'debate_answer': None,
                    'reason': None,
                    'message_history': None,
                    'timestamp': None,
                    'used_head_fallback': None,
                    'embedding_model': None,
                    'file_count': None,
                    'status': 'pending',
                }
                # adr_ids.append(adr_id)
                self.db.cache(value, 'progress')
        self.db.save()
    
    def get_adr_hash(self):
        db_adr_info_path = os.path.join(self.default_path, 'adr_data.db')
        db_adr_info = Database(db_adr_info_path)
        df_adr_info = db_adr_info.to_df('adr_tracking_info')

        df = self.db.to_df('progress')
        df = df[df['extraction_status'] != 'success']

        df_groups = {}

        for index, row in df.iterrows():
            df_adr_info_copy = df_adr_info.copy()
            visited = set()
            first_commit_hash, status, _  = self._get_adr_id(row['commit_hash'], row['adr_name'], df_adr_info_copy, visited)

            value = {
                    'github_address': row['github_address'],
                    'repo_name': row['repo_name'],
                    'adr_name': row['adr_name'],
                    'commit_hash': row['commit_hash'],
                    'first_commit_hash': first_commit_hash,
                    'commit_offset': None,
                    'extraction_status': status,
                    'context_considered_drivers':  row['context_considered_drivers'],
                    'other_sections':  row['other_sections'],
                    'supported_side': None,
                    'debate_answer': None,
                    'reason': None,
                    'message_history': None,
                    'timestamp': None,
                    'used_head_fallback': None,
                    'embedding_model': None,
                    'file_count': None,
                    'status': 'hash_resolved',
                }
            self.db.cache(value, 'progress')
        self.db.save()

    def _get_adr_id(self, commit_hash, adr_name, df_adr_info_copy, visited):
        # if adr_id is not None:
        #     = self._get_first_hash(adr_id, df_adr_info_copy, visited)
        # else:
        #     

        adr_id = commit_hash + '_adr_id_' + adr_name
        if adr_id in df_adr_info_copy['adr_id'].to_list():
            first_commit_hash, status, visited = self._get_first_hash(adr_id, df_adr_info_copy, visited)
            if 'final_commit_hash found' in status:
                return first_commit_hash, status, visited 
        
        if adr_name in df_adr_info_copy['adr_name'].to_list():
            df_adr_info_copy_filtered = df_adr_info_copy[df_adr_info_copy['adr_name'] == adr_name].copy()
            len_df_found = len(df_adr_info_copy_filtered)

            for index in range(len_df_found):
                row_with_newest_timestamp = (
                    df_adr_info_copy_filtered
                    .sort_values('timestamp', ascending=False)
                    .iloc[index]
                )
                new_commit_hash = row_with_newest_timestamp['hash']
                if len(new_commit_hash) > 3: 
                    adr_id = new_commit_hash + '_adr_id_' + adr_name
                    first_commit_hash, status, visited = self._get_first_hash(adr_id, df_adr_info_copy, visited)
                    if 'final_commit_hash found' in status:
                        return first_commit_hash, status, visited  

            first_commit_hash = 'no_commit_hash'
            status = f'found {len_df_found} adr_name, none valid. '
            return first_commit_hash, status, visited 

        else:
            first_commit_hash = 'no_commit_hash'
            status = f'adr_name not found in df_adr_info_copy[adr_name]'
            return first_commit_hash, status, visited 


    def _get_first_hash(self, adr_id, df_adr_info_copy, visited):
        if len(df_adr_info_copy) == 0:
            return 'no_commit_hash', f'initial df_adr_info_copy has len == 0, adr_id: {adr_id}', visited

        if adr_id in visited:
            return 'no_commit_hash', f'adr_id already visited earlier, adr_id: {adr_id}', visited

        df_filtered_adr_id = df_adr_info_copy[df_adr_info_copy['adr_id'] == adr_id]
        if len(df_filtered_adr_id) != 1:
            return 'no_commit_hash', f'adr_id finding error, number of times it was found: {len(df_filtered_adr_id)}. adr_id: {adr_id}', visited

        row_adr_info = df_filtered_adr_id.iloc[0]

        points_to_adr_id = row_adr_info['points_to_adr_id']
        is_oldest = str(row_adr_info['is_oldest'])

        if is_oldest == '1':
            final_commit_hash = str(row_adr_info['hash'])
            status = f'final_commit_hash found {final_commit_hash}'
            return final_commit_hash, status, visited

        elif len(points_to_adr_id) < 3:
            return 'no_commit_hash', f'invalid value for points_to_adr_id: {points_to_adr_id}, {adr_id}', visited

        else:
            visited.add(adr_id)
            return self._get_first_hash(points_to_adr_id, df_adr_info_copy, visited)
            
    
    def extract_data(self, mode='extract', workers=10):
        df = self.db.to_df('progress')
        
        # Check if DataFrame is empty or missing required columns
        if df.empty:
            print(f"No data found in progress table for mode '{mode}'. Please run get_adr_info() first.")
            return
            
        if mode == 'extract':
            if 'extraction_status' not in df.columns or 'first_commit_hash' not in df.columns:
                print("Missing required columns in progress table. Please reinitialize data.")
                return
            df = df[df['extraction_status'] != 'success']
            df = df[df['first_commit_hash'] != 'no_commit_hash']
            # Randomize order to sample across different repos
            df = df.sample(frac=1).reset_index(drop=True)
            print(f"Processing {len(df)} repositories...")
            routine = self.run_extractor
            
        if mode == 'run_mad':
            if 'extraction_status' not in df.columns or 'message_history' not in df.columns:
                print("Missing required columns in progress table for MAD mode.")
                return
            df = df[df['extraction_status'] == 'success']
            # Use isna() to select rows where message_history is NULL/None
            df = df[df['message_history'].isna()]
            # Randomize order to avoid bias
            df = df.sample(frac=1).reset_index(drop=True)
            routine = self.run_mad
            
        futures = {}
        progress_bar_done = tqdm(total=len(df), desc=f"{mode}", position=0, unit=f"run")
        
        # Track file counts for summary
        file_counts = {}
        embedding_models = {}
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for index, row in df.iterrows():
                future = executor.submit(routine, row)
                futures[future] = (row)
                      
            for future in as_completed(futures):
                result = future.result()
                progress_bar_done.update(1)
                
                if result != {}:
                    self.db.cache(result, 'progress')
                    # Track file counts for extract mode
                    if mode == 'extract' and 'repo_name' in result:
                        repo_name = result['repo_name']
                        if repo_name not in file_counts:
                            file_counts[repo_name] = 0
                        # Get file count from the extraction result or database
                        try:
                            # First try to get file count from the result if available
                            if 'file_count' in result:
                                file_counts[repo_name] = result['file_count']
                            else:
                                # Fall back to database query (may fail due to timing/permissions)
                                effective_hash = result.get('first_commit_hash', '')
                                db_path = os.path.join(os.getcwd(), 'dataset', repo_name + effective_hash[:10], 'info.db')
                                if os.path.exists(db_path):
                                    extractor_db = Database(db_path)
                                    files_df = extractor_db.to_df('files_traversed')
                                    file_counts[repo_name] = len(files_df)
                                else:
                                    file_counts[repo_name] = 0  # Database not found
                            
                            if result.get('embedding_model'):
                                embedding_models[repo_name] = result['embedding_model']
                        except Exception as e:
                            print(f"WARNING: Could not compute file count for {repo_name}: {e}")
                            # Set a default value to prevent key errors
                            if repo_name not in file_counts:
                                file_counts[repo_name] = 0
                self.db.save()
                
        progress_bar_done.close()
        
        # Print summary for extract mode
        if mode == 'extract' and file_counts:
            print(f"\n=== VECTORSTORE CREATION SUMMARY ===")
            print(f"Total repositories processed: {len(file_counts)}")
            for repo_name, count in file_counts.items():
                model = embedding_models.get(repo_name, 'unknown')
                print(f"  {repo_name}: {count} files included in vectorstore | embedding: {model}")
            print(f"=====================================\n")        
        
        
    def run_extractor(self, context):
        github_address = context['github_address']
        repo_name = context['repo_name']
        first_commit_hash = context['first_commit_hash']
        if len(first_commit_hash) < 5:
            return {}
        
        extractor = Extractor(github_address, repo_name, first_commit_hash)
        result = extractor.run_routines()
        
        result['adr_name'] = context['adr_name']
        result['context_considered_drivers'] = context['context_considered_drivers']
        result['other_sections'] = context['other_sections']
        result['supported_side'] = None
        result['debate_answer'] = None
        result['commit_hash'] = context['commit_hash']
        result['reason'] = None
        result['message_history'] = None
        result['timestamp'] = datetime.now()
        
        # Add status if not already present
        if 'status' not in result:
            result['status'] = 'extracted'
        
        return result
    
    def run_mad(self, context):
        github_address = context['github_address']
        repo_name = context['repo_name']
        first_commit_hash = context['first_commit_hash']
        
        extractor = Extractor(github_address, repo_name, first_commit_hash, ready=True)
        result = extractor.run_mad(context)
                
        result['adr_name'] = context['adr_name']
        result['github_address'] = context['github_address']
        result['first_commit_hash'] = context['first_commit_hash']
        result['repo_name'] = context['repo_name']
        result['commit_offset'] = context['commit_offset']
        result['commit_hash'] = context['commit_hash']
        result['timestamp'] = datetime.now()
        # Ensure required DB columns are present for upsert
        # Carry over extraction-time fields or set safe defaults
        result['used_head_fallback'] = context.get('used_head_fallback', False)
        result['embedding_model'] = context.get('embedding_model', extractor.retriever_info.get('model'))
        result['file_count'] = context.get('file_count', 0)
        
        # Add status for MAD completion
        if 'status' not in result:
            result['status'] = 'debate_completed'
        
        return result


def main():
    EE = ExtractorEnsambler()
    EE.get_adr_info()  # Initialize data from CSV
    
    # Run full extraction pipeline
    # print("Starting extraction pipeline...")
    # EE.extract_data(mode='extract', workers=10)
    
    # Temporarily commented out for vectorstore focus
    print("Starting multi-agent debate (MAD) pipeline...")
    EE.extract_data(mode='run_mad', workers=1)  # Fewer workers for debate to avoid API rate limits
    
if __name__ == '__main__':
    main()
    
   