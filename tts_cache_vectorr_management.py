import argparse
import os
import re
import sqlite3
from annoy import AnnoyIndex
import numpy as np
from sentence_transformers import SentenceTransformer
from mutagen import MutagenError
from mutagen.id3 import ID3

def validate_cache_entry(title, subtitle, domain):
    """
    Validate a cached MP3 entry against sentence-splitting rules.
    For TTS we validate title (the spoken text).
    For impulse/interject/interrupt/meta we validate subtitle (the generated
    response / phonetic content) because title is user input or a description.
    Returns (is_valid: bool, reason: str).
    """
    sub = str(subtitle).strip() if subtitle else ""

    if domain == "tts":
        text = str(title).strip() if title else ""
    else:
        text = sub
        sub = ""  # already checked as text

    if not text:
        return False, "empty content"
    if text != text.strip():
        return False, "leading/trailing whitespace"
    if '\n' in text or '\r' in text:
        return False, "contains newline"
    # We only care about leading/trailing whitespace (checked above).
    # Internal double-spaces are harmless for TTS and not worth deleting over.
    if 'N/A' in text or 'N/A' in sub:
        return False, "contains N/A"
    if '[' in text or ']' in text or '🎵' in text:
        return False, "contains brackets or music emoji"

    # Only TTS domain gets strict sentence rules.
    # Impulse, interject, interrupt are short generated fillers/responses.
    # Meta is phonetic / onomatopoeic.
    if domain == "tts":
        if '*' in text:
            return False, "contains meta-tag asterisk"
        if not re.search(r'[.!?]+$', text):
            return False, "incomplete sentence (no terminal punctuation)"
        sentences = list(re.finditer(r'[^.!?]*[.!?]+', text))
        if len(sentences) > 1:
            return False, f"multiple sentences ({len(sentences)})"

    return True, "ok"


def clean_domain(directory, domain, dry_run=True):
    """
    Scan a domain directory and remove dud MP3s that don't follow sentence-split rules.
    Returns (kept_count, removed_count, removed_files: list).
    """
    if not os.path.exists(directory):
        print(f"  Directory does not exist: {directory}")
        return 0, 0, []

    kept = 0
    removed = 0
    removed_files = []

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".mp3"):
            continue
        file_path = os.path.join(directory, filename)
        try:
            audio = ID3(file_path)
            title = audio.get('TIT2', [None])[0]
            subtitle = audio.get('TIT3', [None])[0]
        except Exception as e:
            print(f"  [DUD] {filename} — unreadable ID3 tags ({e})")
            removed += 1
            removed_files.append((filename, "unreadable tags"))
            if not dry_run:
                os.remove(file_path)
            continue

        valid, reason = validate_cache_entry(title, subtitle, domain)
        if valid:
            kept += 1
        else:
            print(f"  [DUD] {filename} — {reason} | title={str(title)!r}")
            removed += 1
            removed_files.append((filename, reason))
            if not dry_run:
                os.remove(file_path)

    action = "Would remove" if dry_run else "Removed"
    print(f"  {action} {removed} duds, keeping {kept} valid files.")
    return kept, removed, removed_files


def extract_titles_and_subtitles_from_mp3(directory, remove_duplicates=False, max_duplicates=0):
    """
    Read all MP3 files from a directory and extract their ID3 tags.
    No validation here — cleaning should be done separately via clean_domain().
    """
    data = []
    unique_titles = set()

    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist.")
        return data

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".mp3"):
            try:
                audio = ID3(file_path)
                title = audio.get('TIT2', [None])[0]
                subtitle = audio.get('TIT3', [None])[0]

                if not title:
                    continue

                title = str(title).strip()
                subtitle = str(subtitle).strip() if subtitle else ""

                if remove_duplicates:
                    if title in unique_titles:
                        if len([d for d in data if d[1] == title]) < max_duplicates:
                            data.append((filename, title, subtitle))
                    else:
                        unique_titles.add(title)
                        data.append((filename, title, subtitle))
                else:
                    data.append((filename, title, subtitle))
            except (MutagenError, KeyError):
                pass

    if not data:
        print(f"No MP3 files found in directory {directory}.")
    else:
        print(f"Extracted {len(data)} MP3s from {directory}")

    return data

def generate_embeddings(texts, model):
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    for i, text in enumerate(texts, start=1):
        print(f"Generated embedding for text {i}/{len(texts)}: {text}")
    return list(embeddings)

def create_db(db_path, table_name):
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Deleted existing database file: {db_path}")

    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(f'''CREATE TABLE IF NOT EXISTS {table_name} (filename TEXT PRIMARY KEY, title TEXT, subtitle TEXT, embedding BLOB)''')
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error creating SQLite database: {e}")

def store_embeddings(db_path, table_name, embeddings_data):
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        for filename, title, subtitle, embedding in embeddings_data:
            embedding_blob = embedding.tobytes()
            c.execute(f"INSERT OR REPLACE INTO {table_name} (filename, title, subtitle, embedding) VALUES (?, ?, ?, ?)",
                      (filename, title, subtitle, embedding_blob))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error storing embeddings in SQLite database: {e}")

def create_and_save_annoy_index(db_path, index_path, table_name, f):
    try:
        annoy_index = AnnoyIndex(f, 'angular')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute(f"SELECT filename, embedding FROM {table_name}")
        for i, (filename, embedding_blob) in enumerate(c.fetchall()):
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            annoy_index.add_item(i, embedding)
        annoy_index.build(10)
        annoy_index.save(index_path)
        conn.close()
        print(f"Annoy index saved to {index_path}.")
    except (sqlite3.Error, Exception) as e:
        print(f"Error creating and saving Annoy index: {e}")
    finally:
        if os.path.exists(index_path):
            print(f"File '{index_path}' exists after the process.")

def process_store_and_index(directory, db_path, index_path, table_name, model, remove_duplicates=False, max_duplicates=0):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    create_db(db_path, table_name)
    data = extract_titles_and_subtitles_from_mp3(directory, remove_duplicates, max_duplicates)
    if data:
        embeddings = generate_embeddings([title for _, title, _ in data], model)
        embeddings_data = list(zip([filename for filename, _, _ in data], [title for _, title, _ in data], [subtitle for _, _, subtitle in data], embeddings))
        store_embeddings(db_path, table_name, embeddings_data)
        create_and_save_annoy_index(db_path, index_path, table_name, f=embeddings[0].shape[0])
        print(f"  Indexed {len(data)} embeddings.")
        return len(data)
    else:
        print(f"No data to process from directory {directory}.")
        return 0

def run_clean(dry_run=True):
    AUDIO_ROOT = os.path.join("data", "audio")
    domains = ["tts", "impulse", "meta", "interrupt", "interject"]
    mode = "DRY RUN" if dry_run else "LIVE DELETE"
    print(f"\n=== CACHE CLEANING ({mode}) ===\n")
    total_kept = 0
    total_removed = 0
    for dom in domains:
        dir_path = os.path.join(AUDIO_ROOT, dom)
        print(f"--- Scanning Domain: {dom.upper()} ---")
        kept, removed, _ = clean_domain(dir_path, dom, dry_run=dry_run)
        total_kept += kept
        total_removed += removed
    print(f"\n=== SUMMARY ===")
    print(f"Total kept:   {total_kept}")
    print(f"Total duds:   {total_removed}")
    if dry_run:
        print("Choose option 2 to actually delete these duds.")
    print()


def run_rebuild():
    AUDIO_ROOT = os.path.join("data", "audio")
    VECTOR_ROOT = os.path.join("data", "vector")
    domains = [
        ("tts", "tts_embeddings", 5),
        ("impulse", "impulse_embeddings", 10),
        ("meta", "meta_embeddings", 10),
        ("interrupt", "interrupt_embeddings", 10),
        ("interject", "interject_embeddings", 10)
    ]

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"Model loaded on {model.device}")

    total_embeddings = 0
    per_domain = []
    for dom, table, dups in domains:
        dir_path = os.path.join(AUDIO_ROOT, dom)
        db_path = os.path.join(VECTOR_ROOT, f"{dom}_embeddings.db")
        idx_path = os.path.join(VECTOR_ROOT, f"{dom}_embeddings_index.ann")
        
        print(f"\n--- Processing Domain: {dom.upper()} ---")
        count = process_store_and_index(dir_path, db_path, idx_path, table, model, remove_duplicates=True, max_duplicates=dups)
        total_embeddings += count
        per_domain.append(f"{dom}={count}")

    print(f"\n=== REBUILD SUMMARY ===")
    print(f"Total embeddings: {total_embeddings}")
    print(f"Per domain: {', '.join(per_domain)}")
    print("\nAll domains processed successfully.")


if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("  TTS CACHE & VECTOR MANAGEMENT")
        print("=" * 50)
        print("  1. Dry run — preview duds (no files deleted)")
        print("  2. Clean — delete dud MP3s")
        print("  3. Rebuild vectors — regenerate DBs + indexes")
        print("  4. Clean + Rebuild — delete duds, then rebuild")
        print("  5. Exit")
        print("=" * 50)
        choice = input("\nSelect option: ").strip()

        if choice == "1":
            run_clean(dry_run=True)
        elif choice == "2":
            confirm = input("  This will PERMANENTLY delete files. Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                run_clean(dry_run=False)
            else:
                print("  Cancelled.")
        elif choice == "3":
            run_rebuild()
        elif choice == "4":
            confirm = input("  This will PERMANENTLY delete files. Type 'yes' to confirm: ").strip().lower()
            if confirm == "yes":
                run_clean(dry_run=False)
                run_rebuild()
            else:
                print("  Cancelled.")
        elif choice == "5":
            print("  Exiting.")
            break
        else:
            print("  Invalid option. Try again.")