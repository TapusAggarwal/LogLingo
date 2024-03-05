## Overview
This program is designed to provide comprehensive log data analysis and text processing capabilities. It integrates data extraction, database operations, and text similarity assessments to allow users to interact with and analyze log data effectively.
### Components
- `DataExtraction.py`: This module is responsible for parsing log files, extracting useful information such as unique words, and performing initial data processing tasks. It includes functions for regular expression parsing and can interface with SQLite databases for data storage and retrieval.
- `FunctionCalling.py`: This script enhances the program's capabilities by offering advanced text processing functions, including the computation of TF-IDF vectors and cosine similarity between text entries. It facilitates in-depth analysis of textual data, leveraging machine learning techniques for similarity assessments.
- `app.py`: Serving as the entry point of the application, this script orchestrates user interactions, integrating functionalities from the other modules to process user queries or commands. It's designed to handle complex data analysis requests, providing a user-friendly interface for log data examination and text analysis.

## Setup Instructions
1. Ensure Python 3.x is installed on your system.
2. Install required dependencies:
   ```
   pip install sqlite3 sklearn numpy
   ```
3. Place your log data file in a known directory for processing.
4. Configure database connection settings in `DataExtraction.py` as needed.

## Usage
To use the program, run the `app.py` script from your terminal or command prompt:
```
python app.py
```
Follow the prompts to input your queries or analysis requests. The program supports a variety of operations, from simple log data retrieval to complex text similarity assessments.

## Features
- Log data parsing and unique word extraction
- SQLite database integration for data management
- Advanced text processing with TF-IDF and cosine similarity
- User-friendly command-line interface for interactive data analysis

## Contributing
We welcome contributions and suggestions! Please create an issue or pull request on our GitHub repository to propose changes or additions.

## License
This program is released under the MIT License. See the LICENSE file for more details.
