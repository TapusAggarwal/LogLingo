import datetime
import numpy as np
import autogen
import sqlite3
from typing import Any
from typing_extensions import Annotated
import FunctionCalling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

prompt = input("Prompt: ")
# prompt = "Give all the logs where there is something related to a an error on Nov 08 during the interval 13:11:42 to 13:11:46 both inclusive."
# prompt = "Can you explain what actually happened from log_id 42872 to 42882?"
# prompt = "where did the kernel enabled support for the XSAVE feature 'AVX registers' for its fpu"
# prompt = "where did the function remove_mb_sync_connection was successfully called in the script start_mrt.sh"
# prompt = "Give all the logs where USB hub was found between nov 8 13:11:42 to 13:11:46 both inclusive."
# prompt = "Give all the logs where there is something related to a an error on Nov 08 during the interval 13:11:42 to 13:11:46 both inclusive."
# prompt = "Give all the logs where there is something related to a an error on the server"
# prompt = "Tell instances where .NET thread pool is being used"

prompt += " these are all the keywords found in the database." + str(
    FunctionCalling.get_relevant_keywords(
        prompt))

config_list = [
    [{
        'model': 'gpt-4-turbo-preview',
        'api_key': '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'
    }]
    ,
    [{
        'model': 'gpt-3.5-turbo-1106',
        'api_key': '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'
    }]
]

llm_config4 = {
    "config_list": config_list[0],
    "timeout": 120,
    "temperature": 0
}

llm_config3 = {
    "config_list": config_list[0],
    "timeout": 120,
    "temperature": 0
}


def is_termination_msg(x):
    # print(x)
    return "TERMINATE" in x.get("content", "")


head = autogen.AssistantAgent(
    name="Head",
    # system_message="You are the decision maker if the query is too general to narrow down, then you will ask your advisor what to do next, if it advices you to ask for clarification then you can turn to the human and ask for a human input. If the there are no results for a query than it probably means that sql-writer is writing too specific queries ask for advisor for advice. The preferred range is more than one and less then 100 logs if this is the case then ask the final interpreter to do the final interpretation of the logs.",
    # Keyword Extraction: Leverage the Advisor and the Keywords Finder Bot to identify and confirm the relevant keywords from the user's prompt that are present in our database. This step is crucial for ensuring that subsequent queries are focused and effective.
    system_message=f'''As the Head, your primary role is to orchestrate the analysis and interpretation of complex log data inquiries. Your current task involves dissecting a user prompt that requests information on {prompt}, as indicated by the keywords present in our logs database. Your mission is to lead a series of strategic operations to ensure a comprehensive response to this inquiry. This involves:
First Step: Consult The Advisor for next steps.
Query Formulation and Execution: Once the keywords are identified, instruct the SQLite Writer Bot to craft and execute precise SQLite queries based on these keywords. The goal is to retrieve relevant log entries from the database.
Final Interpretation: If the SQLite Writer Bot successfully retrieves an appropriate number of logs, request the Final Interpreter to analyze and summarize these logs, highlighting instances that directly relate to the user prompt.
Human Input: If the user says to refine the search then repeat the process with the advisor and the SQL_Writer_Bot until you get down to 100 lines. After that call final interpreter to do the final interpretation of the logs.
If the relevant logs are found then give just the logs that are the most relevant to the user prompt and write TERMINATE at the end of your response.
Note: Move Step by Step only one instruction at a time. Give very small instructions to the bots and very small responses.
Throughout this process, maintain clear communication with your team, providing instructions and feedback as needed. If at any point the task seems unfeasible or the query results are not satisfactory, reassess the approach and consult with the Advisor for alternative strategies. Remember, the ultimate goal is to provide the user with a concise, relevant, and informative response regarding {prompt}"''',
    llm_config=llm_config4,
    human_input_mode="TERMINATE",
)

# Advisor to the head
########################################################################################################################
# 2. Command the Keywords_Finder_Bot to extract relevant keywords from the original prompt. For that you need to tell it what is text you want it to find keywords in. At the end of the response mention like this prompt for keywords_finder_bot: followed by the text you want to find the keywords for.
advisor_to_the_head = autogen.AssistantAgent(
    name="advisor_to_the_head",
    system_message=f"""As the Advisor to the Head, your primary role is to critically evaluate incoming queries for their specificity. When a query is deemed too broad or general, you will engage with the SQL_Writer_Bot so it can produce and execute a strategic query. This SQLite query should be designed to extract key information that sheds light on the ambiguous aspects of the original query. Your goal is to refine or suggest a more targeted line of inquiry based on this new information. This process involves:

Step 1. Analysing what kind of queries we need.
There can be two types:
Type 1. Where SQLite queries will suffice and human intervention maybe required if there are more then 100 queries.
Type 2. Where we need to actually analyze the logs for that we may need to look at bunch of logs at the same time to get the context of the logs. And help the user to find the relevant logs. For this you will need to engage with final_interpreter for the final interpretation of the logs. If it successfully finds the logs related to the user query then write TERMINATE at the end of your response with a summary of the found logs. 
Step 2. Requesting the SQL_Writer_Bot to formulate a specific SQLite query (using the keywords given from SQL_Writer_Bot) that can fetch data or insights to clarify these aspects. You do not write the query that is SQL_Writer_Bot's job.
Step 3. Analyzing the results of the SQLite query and advising the Head on the next steps to take.

remember to move step by step detect if a step is complete and then move to the next step. Give very small instructions to the bots and very small responses.

This is the tables available to the SQL_Writer_Bot: 
1. 'logs' with the following columns:
log_id TEXT, time TEXT (always use format {datetime.datetime.now().year}-MM-DD HH:MM:SS year is fixed in out case), layer_source TEXT, message TEXT
The message column contains log messages. Each log message is a string of alphanumeric characters and spaces.
 
Original prompt given by the user:{prompt}

Your inputs will be pivotal in streamlining the decision-making process by ensuring that the Head receives concise, relevant, and actionable information.
Remember to keep the responses small.""",
    llm_config=llm_config4,
    human_input_mode="TERMINATE",
)
########################################################################################################################


# Human input agent
########################################################################################################################
human = autogen.AssistantAgent(
    name="human_input",
    llm_config=llm_config4,
    system_message="Ask for clarification if the head asks for human input. Always use the function you are provided with.",
    is_termination_msg=is_termination_msg,
)


@head.register_for_execution()
@human.register_for_llm(description="Get human input for the head.")
def get_head_decision(msg: Annotated[
    str, "Very short context msg to the human. And short question what is required by the user."]) -> str:
    prompt = input(f"{msg}. You can type 'exit' to end the conversation:")
    if prompt == "exit":
        exit(0)
    return prompt + " these are all the keywords found in logs related to this prompt." + str(
        FunctionCalling.get_relevant_keywords(
            prompt))


########################################################################################################################


# Final interpreter
########################################################################################################################
final_interpreter = autogen.AssistantAgent(
    name="final_interpreter",
    system_message='''You are really good at interpreting logs even with small context.
There are two jobs for you:
Type 1: If the logs are less then 100 and the final task left in the que is to interpret the logs. Then you will interpret the logs and give the final interpretation of the logs in a more Developer understandable language. Remember to Include all the logs that are relevant to the user prompt. And don't forget to include the log_id. After this task you will write TERMINATE at the end of your response.
Type 2: The logs need to be understood for the context of the user prompt or for the advisor to the head. In this case you will report these interpretations to the advisor to the head and the head will decide what to do next. In this case keep the response small and only include the relevant logs and ask the advisor to the head for the next steps.
''',
    llm_config=llm_config3,
    # human_input_mode="ALWAYS"
)
########################################################################################################################

# Sqlite writer bot
########################################################################################################################
sql_writer_bot = autogen.AssistantAgent(
    name="SQLite_writer_bot",
    system_message=f"""
You will write SQLite queries for a log database file on the que of advisor_to_the_head. In the prompt you will also be given list of unique words that are present in the log messages that were extracted based upon the prompt of the user. Remember these are the only words that are present in logs. 
This is the tables available to you: 
1. 'logs' with the following columns:
log_id TEXT, time TEXT (always use format {datetime.datetime.now().year}-MM-DD HH:MM:SS year is fixed in out case), layer_source TEXT, message TEXT
The message column contains log messages. Each log message is a string of alphanumeric characters and spaces.

Original prompt given by the user:{prompt}. Fill in the details if any missed by the advisor.

Use only the functions you have been provided with.
When making function call make sure to use take_feedback_if_less as True if and only if you are sure that all the deductions related to the user query are done. If still there is scope of improvement then always use False.
Don't make too specific queries for example use only the keywords that have a very good chance of being present in the logs. And at max use 3 keywords in a query.
If you do not find any logs then simplify the query, maybe the query is it to specific.
""",
    llm_config=llm_config4,
)


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


@head.register_for_execution()
@sql_writer_bot.register_for_llm(
    description="SQLite query runner. Returns the number of results followed by the results. If take_feedback_if_less is then the user feedback is returned. Now rerun the queries but with the feedback from the user. After one time try to keep take_feedback_if_less False.")
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
                    0], 'The user gave following feedback for the results found, advisor_to_the_head please decide what to do next:', feedback

        return ans

    else:
        return 0, 'Error', [("Error occured", "On funciton call Database connection failed")]


########################################################################################################################

groupchat = autogen.GroupChat(
    agents=[head, advisor_to_the_head, human, sql_writer_bot, final_interpreter],
    messages=[], max_round=12)

manager = autogen.GroupChatManager(groupchat=groupchat,
                                   llm_config=llm_config4,
                                   system_message="You are Group chat Manager.",
                                   is_termination_msg=is_termination_msg,
                                   human_input_mode="TERMINATE")

# start the conversation
manager.initiate_chat(
    head,
    message=prompt,
)
