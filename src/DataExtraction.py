import sqlite3
import re
import json
import os
from datetime import datetime

def extract_unique_words(message):
    words = re.findall(r'\b[a-zA-Z]{2,}\b', message)
    return set(words)

def parse_log_line(line):
    log_pattern = re.compile(r'(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)')
    match = log_pattern.match(line)
    if match:
        time_str, layer_source, message = match.groups()
        # Assuming the log is from the current year
        current_year = datetime.now().year
        # Parsing the datetime with the current year
        time_obj = datetime.strptime(f'{time_str} {current_year}', '%b %d %H:%M:%S %Y')
        return time_obj, layer_source, message
    else:
        return None

def process_log_file_and_save_to_db(file_path, db_path='log_data.db', encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        logs = [parse_log_line(line) for line in file if parse_log_line(line)]

    if logs:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create logs table if it doesn't exist with log_id as PRIMARY KEY
        cursor.execute("CREATE TABLE IF NOT EXISTS logs(log_id INTEGER PRIMARY KEY AUTOINCREMENT, time TEXT, layer_source TEXT, message TEXT)")

        # Insert new logs
        cursor.executemany('INSERT INTO logs (time, layer_source, message) VALUES (?, ?, ?)', logs)
        conn.commit()

        # Fetch all log messages to update word-logID mapping
        cursor.execute('SELECT log_id, message FROM logs')
        all_logs = cursor.fetchall()

        word_logid_mapping = {}
        for log_id, message in all_logs:
            unique_words = extract_unique_words(message)
            for word in unique_words:
                word_logid_mapping.setdefault(word, set()).add(log_id)

        # Create or update word_logid_mapping table
        cursor.execute("CREATE TABLE IF NOT EXISTS word_logid_mapping (word TEXT PRIMARY KEY, log_ids TEXT)")

        for word, log_ids in word_logid_mapping.items():
            json_log_ids = json.dumps(list(log_ids))
            cursor.execute("INSERT OR REPLACE INTO word_logid_mapping (word, log_ids) VALUES (?, ?)",
                           (word, json_log_ids))

        conn.commit()
        conn.close()

        print(f"Processed {len(logs)} log entries and updated the database with word-logID mapping.")

# Assuming __file__ is defined in your context, or you may need to adjust how you define log_file_path
current_directory = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(current_directory, 'test_log2.out')

process_log_file_and_save_to_db(log_file_path)
