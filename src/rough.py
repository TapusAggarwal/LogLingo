import datetime
import numpy as np
import os
import sqlite3
from typing import Literal, List, Tuple, Any
from typing_extensions import Annotated
import FunctionCalling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_different_logs(results):
    # Extract messages from results assuming the message is the last element in each tuple
    messages = [result[-1] for result in results]

    # Check if we need to process the logs further or just return them as they are
    if len(" ".join(messages)) > 7000 or len(messages) > 100:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(messages)

        # Calculate cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix)

        # Invert the similarity scores to get distance matrix
        distance_matrix = 1 - cosine_sim

        # Initialize the selection process
        selected_indices = [0]  # Start with the first log line
        for _ in range(99):  # Since we need 10 unique logs
            last_index = selected_indices[-1]
            # Find the most dissimilar log
            new_index = np.argmax(distance_matrix[last_index])
            while new_index in selected_indices:
                distance_matrix[last_index][new_index] = -1  # Ensure it's not selected again
                new_index = np.argmax(distance_matrix[last_index])
            selected_indices.append(new_index)

        # Prepare selected results based on indices
        selected_results = [results[i] for i in selected_indices]
        total_len = 0

        if len(results) <= 100:
            for i, result in enumerate(selected_results):
                total_len += len(result[1])
                return len(
                    results), 'Results contain more then 10000 tokens so Selected logs based on diversity.', selected_results[
                                                                                                             :i]
        else:
            for i, result in enumerate(selected_results):
                total_len += len(result[-1])
                if total_len > 7000:
                    return len(
                        results), 'Results contain more then 100 lines so Selected logs based on diversity.', selected_results[
                                                                                                              :i]
        return len(results), 'Showing Complete Results', results

    else:
        return len(results), 'Showing Complete Results', results


def query(sqlite: Annotated[str, "SQLite Query"], take_feedback_if_less: Annotated[
    bool, "If this query yields results less then 100, then should we ask the user, what they want to do with this result True will take input from user and false means this query is just for the context of the agents so no input required."]) -> \
        tuple[int, str, list[Any] | str]:
    db_file = r"log_data.db"
    if sqlite[-1] == ";":
        sqlite = sqlite[:-1]
    conn = sqlite3.connect(db_file)
    print("Connected to the SQLite database.")
    if conn is not None:
        try:
            cursor = conn.cursor()
            cursor.execute(sqlite)
            results = cursor.fetchall()
        except sqlite3.Error as err:
            return 0, 'Error', [("Error occured", err)]
        conn.close()
        print(results)
        ans = get_different_logs(results)

        if 100 > ans[0] > 0:
            if take_feedback_if_less:

                feedback_msg = ""

                for i in ans[2]:
                    feedback_msg += str(i) + "\n"

                feedback_msg += "\n" + str(ans[
                                               0]) + ' Results found that match your query do you have some specific query for this set of results? '
                print(feedback_msg, end='')
                feedback = input()
                if feedback == "exit":
                    exit(0)
                return ans[
                    0], 'The user gave following feedback for the results found, SqlBot please decide what to do next:', feedback

        return ans

    else:
        return 0, 'Error', [("Error occured", "On funciton call Database connection failed")]


print(query(
    "SELECT * FROM logs WHERE message LIKE '%kernel%' AND message LIKE '%AVX%' ORDER BY time ASC",
    False))
