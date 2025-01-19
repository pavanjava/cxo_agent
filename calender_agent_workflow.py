from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step, Context
)
from composio_llamaindex import App, ComposioToolSet
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from datetime import datetime
from llama_index.utils.workflow import draw_all_possible_flows
from events import PrefixMessageEvent, AgentEvent
import dotenv
import openai
import json

# Load environment variables from .env file
dotenv.load_dotenv()


class CalenderAgenticWorkflow(Workflow):
    @step
    async def initialize(self, ev: StartEvent, ctx: Context) -> PrefixMessageEvent:
        # Initialize the LLM
        llm = OpenAI(model="gpt-4o")

        # Initialize the ComposioToolSet
        toolset = ComposioToolSet()

        # Get the RAG tool from the Composio ToolSet
        tools = toolset.get_tools(apps=[App.GOOGLECALENDAR])

        # Define the prefix message for Agent
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

        await ctx.set("query", ev.query)
        await ctx.set("tools", tools)
        await ctx.set("llm", llm)

        return PrefixMessageEvent(prefix_messages=prefix_messages)

    @step
    async def create_agent(self, ev: PrefixMessageEvent, ctx: Context) -> AgentEvent:
        tools = await ctx.get("tools")
        llm = await ctx.get("llm")
        prefix_messages = ev.prefix_messages

        # Initialize a FunctionCallingAgentWorker with the tools, LLM, and system messages
        agent = FunctionCallingAgentWorker(
            tools=tools,  # Tools available for the agent to use
            llm=llm,  # Language model for processing requests
            prefix_messages=prefix_messages,  # Initial system messages for context
            max_function_calls=10,  # Maximum number of function calls allowed
            allow_parallel_tool_calls=False,  # Disallow parallel tool calls
            verbose=True,  # Enable verbose output
        ).as_agent()

        return AgentEvent(agent=agent)

    @step
    async def run_agent(self, ev: AgentEvent, ctx: Context) -> StopEvent:
        agent = ev.agent
        query = await ctx.get("query")
        response = agent.chat(query)
        return StopEvent(result=response)


def get_details_from_promot(user_prompt: str):
    system_prompt = """
    As a personal assistant, analyze the given prompt and extract the following information:
    - date: Convert any relative or absolute dates to YYYY-MM-DD format
    - timezone: Extract any mentioned timezone example: IST, CST(leave empty if not specified)
    - timeslot: Format time range as "HH:MM AM/PM - HH:MM AM/PM"
    - intent: Determine if the action is create_meeting with xyz person, create_meeting on topic, cancel_meeting, or reschedule_meeting
    
    Return the information in the following JSON structure, maintaining the exact format:
    {
        "date": "YYYY-MM-DD",
        "timezone": "IST",
        "timeslot": "HH:MM AM/PM - HH:MM AM/PM",
        "intent": "meeting with xyz person or meeting on specific topic etc"
    }
    Note: Always get the today's date"""+str(datetime.today())+"""and formulate the target dates based on this.
    """

    # Send the preprocessed prompt to the LLM
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # Extract the response
    llm_response = json.loads(response.choices[0].message.content)
    print(llm_response)
    return llm_response


async def main():
    response = get_details_from_promot("book a slot for a meeting with keshav tomorrow from 3.00 PM to 4.00 PM on topic ai agents discussion")
    task = f"""
                Book meeting slots according to "{response['timeslot']} -> {response['intent']}".
                Properly label them with the work provided to be don e in that time period.
                Schedule it for today. Today's date is {response['date']} (it's in YYYY-MM-DD format)
                and make the timezone be {response['timezone']}.
                """

    print(task)
    w = CalenderAgenticWorkflow(timeout=100, verbose=True)
    result = await w.run(query=task)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

