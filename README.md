
# Q&A chatbot for the openpowerlifting database

This project provides a Python-based agent that leverages the LangChain framework to interact with a PostgreSQL database containing powerlifting competition results. The agent can answer various queries related to powerlifting data by executing SQL queries, processing results, and providing accurate and meaningful responses.

For a step by step guide on how to run the code, including setting up the database, please feel free to check out the article I made about this project.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [Environment Variables](#environment-variables)
  - [Database Connection](#database-connection)
  - [Running the Agent](#running-the-agent)
- [Agent Capabilities](#agent-capabilities)
  - [Available Tools](#available-tools)
  - [Example Queries](#example-queries)


## Introduction

This project integrates the LangChain framework with a PostgreSQL database to create a conversational agent capable of querying powerlifting competition data. The agent is built using OpenAI's API to call the latest gpt model and is designed to answer specific queries related to lifters' performance, competition details, and other related data points stored in the database.

## Features

- **SQL Database Interaction:** Connects to a PostgreSQL database and executes SQL queries.
- **Customizable Agent:** Easily extendable to include additional tools and functionalities.
- **Conversational Memory:** Retains conversation history to provide context-aware responses.
- **Data Integrity:** Ensures accurate responses by handling null values and filtering duplicate results.

## Setup

### Requirements

- Python 3.7 or higher
- PostgreSQL database with relevant powerlifting data
- OpenAI API key

### Installation

1. **Clone the repository:**

    ```
    git clone https://github.com/tonymoszuti/openpowerlifting
    cd openpowerlifting
    ```
    
2. **Install dependencies using Poetry:**

   If you don’t have Poetry installed, you can install it by following the instructions on their website: [https://python-poetry.org/docs](https://python-poetry.org/docs)

    ```
    curl -sSL https://install.python-poetry.org | python3 -
    ```

   Followed by:

    ```
    poetry install
    ```

## Usage

### Environment Variables

You need to set your OpenAI API key in the environment variables:

```
export OPENAI_API_KEY="your-openai-api-key"
```

### Database Connection

Update the database connection details in the code:

```
port = "xxxx"
password = "xxxx"
user = "xxxx"
host = "postgres-db"
dbname = "openpowerlifting"
```
    
You can test if the database connection works by inputting the same connection details into db_connect_test.py and running the code.

### Running the agent

To start the agent and run a query:

```
response = agent.run("What is the best bench press performed by a female lifter?")
print(response)
```

This will output the relevant data from the database.


## Agent Capabilities

### Available Tools

The agent is equipped with several tools that allow it to perform various tasks:

- **Get Distinct Values:** Fetches distinct values from specified columns.
- **Get Today's Date:** Returns the current date.
- **Get Column Descriptions:** Provides descriptions for each column in the powerlifting results table.
- **Fetch Strongest Lifter by Weight Class:** Retrieves the strongest lifter in a specified weight class.

### Example Queries

- "What is the best squat performed by a lifter in the 74kg weight class?"
- "List all distinct weight classes in the database."
- "What is the GL score for the top female lifter?"




Create a .vscode folder and a launch.json file in there with this code:

{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Open Powerlifting SQL Agent",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/python/src/openpowerlifting/main.py",
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "OPENAI_API_KEY": ""
            }
        },
    ]
}