import sqlite3
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def query(sql):
    db_file = r"log_data.db"

    conn = sqlite3.connect(db_file)
    print("Connected to the SQLite database.")
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
        except sqlite3.Error as err:
            return 0, [("Error occured", "On funciton call Database connection failed")]
        conn.close()
        return len(results), results
    else:
        return 0, [("Error occured", "On funciton call Database connection failed")]


def get_relevant_keywords(prompt):
    log_messages = []
    for row in query("SELECT word FROM word_logid_mapping")[1]:
        log_messages.append(row[0])

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(log_messages + [prompt])

    # Calculate cosine similarity between the prompt and each log message
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Get indices of log messages sorted by similarity
    sorted_indices = similarity_scores.argsort()[0][::-1]

    # Extract keywords from top similar log messages
    relevant_keywords = {}

    # Get TF-IDF scores for the prompt
    prompt_tfidf_scores = tfidf_matrix[-1]

    for index in sorted_indices:
        message = log_messages[index]
        words = re.findall(r'\b[a-zA-Z]{2,}\b', message)
        for word in words:
            # Get the index of the word in the TF-IDF matrix
            keyword_index = tfidf_vectorizer.vocabulary_.get(word)
            if keyword_index is not None:
                # Get TF-IDF score for the keyword in the prompt
                keyword_tfidf_score = prompt_tfidf_scores[0, keyword_index]
                # Store the TF-IDF score of the keyword
                relevant_keywords[word] = keyword_tfidf_score

    # Sort keywords by TF-IDF score in descending order
    relevant_keywords = dict(sorted(relevant_keywords.items(), key=lambda item: item[1], reverse=True))

    # Return the top 25 keywords with the highest TF-IDF scores
    return list(relevant_keywords.keys())[:25]
