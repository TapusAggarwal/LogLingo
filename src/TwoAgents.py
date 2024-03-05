import datetime

import autogen
import os
import sqlite3

from typing import Literal, List, Tuple
from typing_extensions import Annotated

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

config_list = [
    {
        'model': 'gpt-4',
        'api_key': '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'
    }
    # ,
    # {
    #     'model': 'gpt-3.5-turbo-0613',
    #     'api_key': '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'
    # }
]

llm_config = {
    "config_list": config_list,
    "timeout": 120,
}

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    # is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").split("\n")[-1],
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "code",
        "use_docker": False
    },
    system_message="You will guide the sql writer to successfully narrow down the logs that user is looking for below 100. If you think that the query is too general to narrow down the search then you can ask the user for more specific details but first read the output that was generated by the assistant and based upon that ask the user for more specific details."
)

# Sqlite writer bot
########################################################################################################################
sql_writer_bot = autogen.AssistantAgent(
    name="SQLite_writer_bot",
    system_message=f"""
You will write SQLite queries for a log database file. In the prompt you will also be given list of unique words that are present in the log messages that were extracted based upon the prompt of the user. Remember these are the only words that are present in logs. 
The file contains two tables 
1. 'logs' with the following columns:
log_id TEXT, time TEXT (always use format {datetime.datetime.now().year}-MM-DD HH:MM:SS), layer_source TEXT, message TEXT
The message column contains log messages. Each log message is a string of alphanumeric characters and spaces.
2. 'word_logid_mapping' with the following columns:
word TEXT PRIMARY KEY, log_ids TEXT
word column contains unique words extracted from the 'message' column of the 'logs' table and 'log_ids' column contains the row indexes of the logs that contain that word.
Use only the functions you have been provided with. Reply with TERMINATE at the end of your response if the task is done.
You're task is to narrow logs to less than 100 logs based on the prompt and then return the logs.
""",
    llm_config=llm_config,
)


# Main function to run a query
@user_proxy.register_for_execution()
@sql_writer_bot.register_for_llm(
    description="SQLite query runner. Returns the number of results and the results. If the number is negative that means the number of results is more than 100 and only the first 100 results are returned. The negative number is the number of results.")
def query(sql: Annotated[str, "SQLite Query"]) -> Tuple[int, List[Tuple]]:
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
        if len(results) > 100:
            return -len(results), results[:100]
        return len(results), results
    else:
        return 0, [("Error occured", "On funciton call Database connection failed")]


########################################################################################################################


#
########################################################################################################################
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


########################################################################################################################

prompt = "Give all the logs where there is something related to a an error on Nov 08 during the interval 13:11:42 to 13:11:46 both inclusive."
# prompt = "Give all the logs where there is something related to a an error on the server"
import FunctionCalling

prompt += " these are all the keywords found in logs related to this prompt." + str(
    FunctionCalling.get_relevant_keywords(
        prompt))

# start the conversation
user_proxy.initiate_chat(
    sql_writer_bot,
    message=prompt,
)

# user_proxy.initiate_chat(
#     sql_writer_bot,
#     message="Give all the logs that where Some Server error was found 13:11:42 to 13:11:46 both inclusive.",
# )