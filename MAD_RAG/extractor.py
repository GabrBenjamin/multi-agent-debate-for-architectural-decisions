
import os
from database import Database
import os
import pandas as pd

from git import Repo
import traceback

import numpy as np  
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from debate_manager import main_debate

class Extractor():
    def __init__(self, github_address, repo_name, commit_hash, ready=False):
        self.offset = 3
        self.repo_name = repo_name.replace('.','').replace('-','')
        self.original_commit_hash = commit_hash
        self.effective_commit_hash = commit_hash
        self.repo_path = os.path.join(os.getcwd(), 'dataset', self.repo_name+commit_hash[:10])
        db_path = os.path.join(self.repo_path, 'info.db')
        self.db = Database(db_path)
        self.repo_clone_path = os.path.join(self.repo_path, self.repo_name)
        self.github_address = github_address
        self.used_head_fallback = False
        self.retriever_info = {
            "collection_name": f"collection_{self.repo_name}",
            "model": "text-embedding-3-large",
            "persist_directory": os.path.join(self.repo_path, f'{self.repo_name}_chroma_db'),
        }
        
        if os.path.exists(self.repo_path) and not ready:
            shutil.rmtree(self.repo_path)
        
        if os.path.exists(self.repo_path) and not ready:
            print(f'ERROR: not erased correctly')
            
        embeddings = OpenAIEmbeddings(model=self.retriever_info["model"])
        self.vector_store = Chroma(
            collection_name=self.retriever_info["collection_name"],
            embedding_function=embeddings,
            persist_directory=self.retriever_info["persist_directory"], 
        )
        
        self.create_tables()

    def create_tables(self):
        self.db.execute_sql("""
        CREATE TABLE IF NOT EXISTS files_traversed (
            status TEXT,
            blob_type TEXT,
            repo_name TEXT,
            file_path TEXT,
            file_name TEXT,
            file_content TEXT,
            PRIMARY KEY (blob_type, repo_name, file_path)
        );
        """)
        
        self.db.execute_sql("""
        CREATE TABLE IF NOT EXISTS test_vectorstore (
            test_type TEXT,
            input TEXT,
            output TEXT,
            status TEXT,
            PRIMARY KEY (test_type, input)
        );
        """)
        
    
    def run_routines(self):
        try:
            repo = self.clone_project()
            # Only collect relevant files for factor extraction (docs/ADRs)
            file_contents = self.get_commit_files(repo, filter_relevant=True)
            status = self.create_vectorstore(file_contents)
            self.test_vectorstore()
            status_suffix = " (used HEAD fallback)" if getattr(self, 'used_head_fallback', False) else ""
            print(f"SUCCESS: {self.repo_name} - {len(file_contents)} files processed{status_suffix}")
            status_message = "success"
            if getattr(self, 'used_head_fallback', False):
                status_message = "success (used HEAD fallback)"
            
            output = {
                'github_address': self.github_address,
                'repo_name': self.repo_name,
                'first_commit_hash': self.effective_commit_hash,
                'extraction_status': status,
                'commit_offset': self.offset,
                'used_head_fallback': getattr(self, 'used_head_fallback', False),
                'embedding_model': self.retriever_info["model"],
                'file_count': len(file_contents),
                'status': status_message,
            }
            return output
        
        except Exception as e:
            print(f'ERROR: {self.repo_name} - {e}')
            error_message = str(e)
            output = {
                'github_address': self.github_address,
                'repo_name': self.repo_name,
                'first_commit_hash': self.effective_commit_hash,
                'extraction_status': f'exception {e}',
                'commit_offset': self.offset,
                'used_head_fallback': False,
                'embedding_model': self.retriever_info["model"],
                'file_count': 0,
                'status': f'error: {error_message}',
            }
            return output
    
            
    def clone_project(self):    
        try:
            print(f"Cloning {self.github_address} to {self.repo_clone_path}")
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.repo_clone_path), exist_ok=True)
            
            # First, test if the repository is accessible
            import subprocess
            try:
                result = subprocess.run(
                    ['git', 'ls-remote', '--heads', self.github_address], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                if result.returncode != 0:
                    print(f"ERROR: Repository {self.github_address} is not accessible")
                    raise Exception(f"Repository not accessible: {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"ERROR: Repository {self.github_address} timed out on access test")
                raise Exception("Repository access timed out")
            
            # Initialize an empty bare repo and fetch only what's needed (be inclusive across branches/tags)
            try:
                # git init --bare
                subprocess.run(['git', 'init', '--bare', self.repo_clone_path], check=True, capture_output=True)
                # git remote add origin <url>
                subprocess.run(['git', '-C', self.repo_clone_path, 'remote', 'add', 'origin', self.github_address], check=True, capture_output=True)
                
                # Try to fetch the specific commit with shallow depth and partial objects
                fetch_specific = subprocess.run([
                    'git', '-C', self.repo_clone_path, 'fetch', '--depth=1', '--filter=blob:none', 'origin', self.effective_commit_hash
                ], capture_output=True, text=True, timeout=180)
                
                if fetch_specific.returncode != 0:
                    # If that fails, fetch all heads shallow (partial clone)
                    fetch_heads = subprocess.run([
                        'git', '-C', self.repo_clone_path, 'fetch', '--depth=100', '--filter=blob:none', 'origin', '+refs/heads/*:refs/heads/*'
                    ], capture_output=True, text=True, timeout=300)
                    # Also fetch tags (may reference historical points)
                    fetch_tags = subprocess.run([
                        'git', '-C', self.repo_clone_path, 'fetch', '--filter=blob:none', '--tags', 'origin'
                    ], capture_output=True, text=True, timeout=180)
                    # Fetch PR refs (common on GitHub)
                    fetch_prs = subprocess.run([
                        'git', '-C', self.repo_clone_path, 'fetch', '--depth=100', '--filter=blob:none', 'origin', '+refs/pull/*:refs/pull/*'
                    ], capture_output=True, text=True, timeout=300)

                    # Retry fetching the specific commit with partial objects
                    retry_specific = subprocess.run([
                        'git', '-C', self.repo_clone_path, 'fetch', '--filter=blob:none', 'origin', self.effective_commit_hash
                    ], capture_output=True, text=True, timeout=300)

                    if retry_specific.returncode != 0 and fetch_heads.returncode != 0:
                        # As last resort, fetch everything (can be slow) with partial objects first
                        fetch_all_partial = subprocess.run([
                            'git', '-C', self.repo_clone_path, 'fetch', '--filter=blob:none', 'origin', '--prune'
                        ], capture_output=True, text=True, timeout=600)
                        if fetch_all_partial.returncode != 0:
                            fetch_all = subprocess.run([
                                'git', '-C', self.repo_clone_path, 'fetch', 'origin', '--prune'
                            ], capture_output=True, text=True, timeout=900)
                            if fetch_all.returncode != 0:
                                raise Exception(f"Git fetch failed: {fetch_specific.stderr or fetch_heads.stderr or fetch_all.stderr}")
                
                repo = Repo(self.repo_clone_path)
                
                # Validate the commit exists now; if not, recompute introducer by ADR path if possible, else fall back to HEAD
                try:
                    repo.commit(self.effective_commit_hash)
                except Exception as e:
                    # Try to locate introducer commit by ADR path if we have an adr_name in context
                    adr_path = None
                    try:
                        # Attempt to infer ADR path from repo-local DB cache if present in progress row
                        # Not available directly here without context; fall back to HEAD
                        pass
                    except Exception:
                        pass
                    head_sha = repo.head.commit.hexsha
                    print(f"WARNING: Commit {self.effective_commit_hash} not found after fetch; falling back to HEAD {head_sha[:8]}.")
                    self.effective_commit_hash = head_sha
                    self.used_head_fallback = True
                
                return repo
            except subprocess.TimeoutExpired:
                print(f"ERROR: Fetch timed out for {self.github_address}")
                raise Exception("Git fetch timed out")
                
        except Exception as e:
            print(f"ERROR: Failed to clone {self.github_address}: {e}")
            raise
    
    def _get_commit_at_offset(self, repo):
        try:
            commit = repo.commit(self.effective_commit_hash)
            # If we fell back to HEAD, avoid stepping back further to reduce information loss
            steps = 0 if getattr(self, 'used_head_fallback', False) else self.offset
            current_level = 0
            while current_level < steps and commit.parents:
                # Choose the most recent parent (to stay as close as possible while stepping back)
                if len(commit.parents) == 1:
                    commit = commit.parents[0]
                else:
                    p = commit.parents
                    # Pick parent with max authored_date to avoid jumping too far back on merges
                    commit = max(p, key=lambda c: getattr(c, 'authored_date', 0))
                current_level += 1
            if current_level < self.offset:
                print(f"WARNING: Requested offset {self.offset} exceeds available history; using oldest reachable commit at offset {current_level}.")
            return commit
        except Exception as e:
            print(f"ERROR: Could not resolve commit {self.effective_commit_hash}: {e}")
            raise
    
    def get_commit_files(self, repo, filter_relevant=False):
        file_contents = {}
        
        target_commit = self._get_commit_at_offset(repo)
        tree = target_commit.tree

        for blob in tree.traverse():
            if filter_relevant and not self._is_relevant_file(blob.path):
                continue
            if blob.type == 'blob':
                file_path = blob.path
                file_name = os.path.basename(file_path)
                try:
                    status = True
                    content = blob.data_stream.read().decode('utf-8')
                    splitted_documents = self.load_filecontent(file_path, content, filter_relevant=filter_relevant)
                    if splitted_documents != []:
                        file_contents[file_path] = splitted_documents
                    
                except UnicodeDecodeError:
                    status = False
                    content = None
                    
            else:
                status = False
                file_path = None 
                file_name = None  
                content = None
                
            info = {
                'blob_type': blob.type,
                'repo_name': self.repo_name,
                'file_path': file_path,
                'file_name': file_name,
                'file_content': content,
                'status': status,
            }
            self.db.cache(info, 'files_traversed')
        self.db.save()

        # If this is a merge commit, include relevant files from the other parents' trees as well (more inclusive)
        if len(target_commit.parents) > 1:
            for parent_commit in target_commit.parents:
                if parent_commit == target_commit.parents[0]:
                    continue
                parent_tree = parent_commit.tree
                for blob in parent_tree.traverse():
                    if filter_relevant and not self._is_relevant_file(blob.path):
                        continue
                    if blob.type == 'blob':
                        file_path = blob.path
                        file_name = os.path.basename(file_path)
                        try:
                            status = True
                            content = blob.data_stream.read().decode('utf-8')
                            splitted_documents = self.load_filecontent(file_path, content, filter_relevant=filter_relevant)
                            if splitted_documents != []:
                                # Do not overwrite if already collected from main tree
                                if file_path not in file_contents:
                                    file_contents[file_path] = splitted_documents
                        except UnicodeDecodeError:
                            status = False
                            content = None
                    else:
                        status = False
                        file_path = None 
                        file_name = None  
                        content = None
                    info = {
                        'blob_type': blob.type,
                        'repo_name': self.repo_name,
                        'file_path': file_path,
                        'file_name': file_name,
                        'file_content': content,
                        'status': status,
                    }
                    self.db.cache(info, 'files_traversed')
            self.db.save()
        return file_contents

    def load_filecontent(self, path, content, filter_relevant=False):
        # For now, rely primarily on file-level filtering. If we later add
        # content-level criteria (e.g., specific sections), we can gate here.
        
        doc = Document(
            page_content=content,
            metadata={"path": path}
        )
         
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = splitter.split_documents([doc])
        return docs

    def create_vectorstore(self, file_contents, filter_relevant=False):  
        try:
            for key, value in file_contents.items():
                self.vector_store.add_documents(documents=value)
            return 'success'
        
        except Exception as e:
            return f'fail, Exception: {e}'

    def test_vectorstore(self):
        test_cases = [
            'adrs',
            'decisions',
            'code',
            'architecture',
            'register',
            'python',
            'database',
        ]
        
        for test_case in test_cases:
            retrieved_docs = self.vector_store.similarity_search(test_case)
            
            test_type = 'similarity_search'
            query = test_case
            output = str(retrieved_docs)
            status = 'success'
            
            result = {
                'test_type': test_type,
                'input': query,
                'output': output,
                'status': status,
            }
            self.db.cache(result, 'test_vectorstore')
        self.db.save()
            
    def run_mad(self, context):
        output = main_debate(context, self.retriever_info)
        return output

    # Helper: determine if a file should be included in the vector store for factor extraction
    def _is_relevant_file(self, file_path: str) -> bool:
        lower_path = file_path.lower()
        base_name = os.path.basename(lower_path)

        # Skip binary and non-text files
        binary_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', '.tar', '.gz', '.exe', '.bin', '.dll', '.so'}
        if any(lower_path.endswith(ext) for ext in binary_extensions):
            return False

        # Direct documentation files (more inclusive)
        if base_name in {"readme.md", "readme.txt", "readme.rst", "contributing.md", "code_of_conduct.md", 
                        "license.md", "license.txt", "authors.md", "install.md", "setup.md"}:
            return True

        # Changelog and history files
        if base_name.startswith(("changelog", "history", "news", "release")):
            return True

        # ADRs and documentation directories (more inclusive)
        doc_patterns = ["/docs/", "/doc/", "/adr/", "/adrs/", "/architecture/", "/design/", 
                       "/documentation/", "/wiki/", "/guides/", "/tutorial/", "/tutorials/",
                       "/decisions/", "/decision/", "/spec/", "/specs/", "/reference/"]
        if any(pattern in lower_path for pattern in doc_patterns):
            return lower_path.endswith((".md", ".txt", ".rst", ".asciidoc"))

        # .github configs and templates
        if "/.github/" in lower_path:
            return lower_path.endswith((".md", ".txt", ".yml", ".yaml"))

        # Root level markdown files (often documentation)
        if "/" not in lower_path.strip("/") and lower_path.endswith(".md"):
            return True

        # Files with documentation-related names anywhere
        doc_keywords = ["adr", "decision", "architecture", "design", "guide", "tutorial", 
                       "spec", "specification", "manual", "handbook"]
        if any(keyword in base_name for keyword in doc_keywords):
            return lower_path.endswith((".md", ".txt", ".rst"))

        return False