import streamlit as st
import dotenv
from composio_llamaindex import App, ComposioToolSet
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from datetime import datetime
from llama_index.core import Settings

# Load environment variables from .env file
dotenv.load_dotenv()
Settings.llm = OpenAI(model="gpt-4o")
llm = OpenAI(model="gpt-4o")

# Initialize the ComposioToolSet
toolset = ComposioToolSet()

# Get the RAG tool from the Composio ToolSet
tools = toolset.get_tools(apps=[App.GOOGLECALENDAR])

# Streamlit UI
st.title("Google Calendar Scheduling Agent")

# Input fields for the user
st.subheader("Provide Details for Scheduling")
todo = st.text_area("Enter your schedule details (e.g., 1:30:00 AM - 2.30:00 AM - Meeting with Aakash):")
timezone = st.text_input("Enter your timezone (e.g., Asia/Kolkata):", value=datetime.now().astimezone().tzinfo)
date = st.date_input("Select the date for scheduling:", value=datetime.today())


# Button to execute the action
if st.button("Book Slots"):
    # Define the RAG Agent
    prefix_messages = [
        ChatMessage(
            role="system",
            content=(
                """
                You are an AI agent responsible for taking actions on Google Calendar on users' behalf. 
                You need to take action on Calendar using Google Calendar APIs. Use correct tools to run APIs from the given tool-set.
                """
            ),
        )
    ]

    # Initialize a FunctionCallingAgentWorker with the tools, LLM, and system messages
    agent = FunctionCallingAgentWorker(
        tools=tools,  # Tools available for the agent to use
        llm=llm,  # Language model for processing requests
        prefix_messages=prefix_messages,  # Initial system messages for context
        max_function_calls=10,  # Maximum number of function calls allowed
        allow_parallel_tool_calls=False,  # Disallow parallel tool calls
        verbose=True,  # Enable verbose output
    ).as_agent()

    # Convert date and timezone to string for the agent
    date_str = date.strftime("%Y-%m-%d")

    # {todo.split("-")[0]} to {todo.split("-")[1]} on topic {todo.split("-")[2]}
    response = agent.chat(
        f"""
        Book slots from {todo.split("-")[0]} to {todo.split("-")[1]} on topic {todo.split("-")[2]}. 
        Properly label them with the work provided to be done in that time period. 
        Schedule it for today. Today's date is {date_str} (it's in YYYY-MM-DD format) 
        and make the timezone be {timezone}.
        """
    )

    # Display the response
    st.subheader("Response from the Agent")
    st.write(response)
