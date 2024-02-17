# TODO Assistant 📝🤖
⚡ Handle Your Notion TODO board with LLMs ⚡

 <p align="center">
    <a href="https://github.com/kosciukiewicz/todo-assistant/actions/workflows/static_analysis.yml">
      <img src="https://github.com/kosciukiewicz/todo-assistant/actions/workflows/static_analysis.yml/badge.svg" />
    </a>
    <a href="https://github.com/kosciukiewicz/protfolio-page/issues">
      <img src="https://img.shields.io/github/issues/kosciukiewicz/todo-assistant"/>
    </a>
    <a href="https://github.com/kosciukiewicz/todo-assistant/stargazers">
      <img src="https://img.shields.io/github/stars/kosciukiewicz/todo-assistant"/>
    </a>
    <a href="https://github.com/kosciukiewicz/todo-assistant/network/members">
      <img src="https://img.shields.io/github/forks/kosciukiewicz/todo-assistant"/>
    </a>
    <a href="https://github.com/kosciukiewicz/todo-assistant/commits/master">
      <img src="https://img.shields.io/github/last-commit/kosciukiewicz/todo-assistant/master"/>
    </a>
  </p>

## Overview 👀

Notion TODO Board Assistant is an AI-powered productivity tool, which leverages the capabilities of LangChain and LangGraph. The application helps to manage your tasks on a Notion board by actively listening to your commands and executing requested tasks. It interfaces with the Notion API to interact with real Notion database. 

A great way to familiarize with the LangChain library trying to make the solution as production ready as possible with a high quality codebase. 🚀

## Features ✨

You can perform an array of actions through commands with TO-DO Assistant:

1. Add a new task.
2. Update existing tasks.
3. Delete tasks.
4. Retrieve specific tasks.
5. Generate tasks based on project description.

And much more!

## Tech Stack 🛠

- LangChain: For LLMs flows.
- LangGraph: For complex agents definition.
- notion-client: For interfacing with Notion database.
- Streamlit: For interactive web application example.

## Installation 💻

Installing and setting up the project is a breeze, all thanks to Poetry:

1. Use poetry to set up the environment and install dependencies:

   `poetry install`

2. Set the env variables. Create the `.env` file based on `.env.example` and fill it with appropriate values

## Running The Project 🚀

### Using Run Script

You can run the assistant in commandline using the provided run.py script:

`python run.py`

### Using Streamlit App

The Streamlit application provides a more interactive experience. Run it using:

`streamlit run streamlit_app.py`

## Example Commands 🎤

Here are a few examples of what you can do:

- "Create a task titled 'Buy Groceries'"
- "Mark task 'Buy Groceries' as Done"
- "Delete the task 'Buy Groceries'"
- "I want to learn python, define and add a few example tasks to the board."
- "What is the status of 'Buy Groceries'"

## Roadmap

- ~~Streamlit application~~
- ~~Saving conversation history~~
- Error handling;
- Handle tasks properties like priority or work estimation;
- Question answering based on the whole database not only the specific retrieved task;

