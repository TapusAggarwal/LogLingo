import sqlite3
import json
import re
from nltk.corpus import wordnet as wn


def extract_unique_words(message):
    words = re.findall(r'\b[a-zA-Z]{2,}\b', message)
    return set(words)


def get_semantically_similar_words(word):
    similar_words = set()
    synsets = wn.synsets(word)
    for synset in synsets:
        hypernyms = synset.hypernyms()
        hyponyms = synset.hyponyms()
        for hypernym in hypernyms:
            similar_words.update(hypernym.lemma_names())
        for hyponym in hyponyms:
            similar_words.update(hyponym.lemma_names())
    return similar_words


def create_word_similar_mapping(db_path='similar.db', log_db_path='log_data.db'):
    conn_log = sqlite3.connect(log_db_path)
    cursor_log = conn_log.cursor()
    conn_similar = sqlite3.connect(db_path)
    cursor_similar = conn_similar.cursor()

    # Ensure the table for word_similar_mapping exists
    cursor_similar.execute('''
    CREATE TABLE IF NOT EXISTS word_similar_mapping (
        word TEXT PRIMARY KEY,
        similar_words TEXT
    )
    ''')

    cursor_log.execute('SELECT message FROM logs')
    log_messages = cursor_log.fetchall()

    word_pool = set()  # Pool of words extracted from log messages

    for message in log_messages:
        unique_words = extract_unique_words(message[0])  # message is a tuple, message[0] contains the text
        word_pool.update(unique_words)

    word_similar_mapping = {}

    for word in word_pool:
        similar_words = get_semantically_similar_words(word)
        similar_words_in_pool = similar_words.intersection(word_pool)
        word_similar_mapping[word] = list(similar_words_in_pool)

    for word, similar_words in word_similar_mapping.items():
        json_similar_words = json.dumps(similar_words)
        cursor_similar.execute('''
            INSERT INTO word_similar_mapping (word, similar_words) VALUES (?, ?)
            ON CONFLICT(word) DO UPDATE SET similar_words=excluded.similar_words
        ''', (word, json_similar_words))

    conn_similar.commit()
    conn_similar.close()
    conn_log.close()


create_word_similar_mapping()

print(f"Word-semantically similar mapping saved to the new database 'similar.db'")
