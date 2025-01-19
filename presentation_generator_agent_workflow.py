from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step, Context
)
from composio_llamaindex import ComposioToolSet, Action
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from events import PrefixMessageEvent, AgentEvent
import dotenv
import json
import openai

# Load environment variables from .env file
dotenv.load_dotenv()


class PresentationGenerationWorkflow(Workflow):
    @step
    async def initialize(self, ev: StartEvent, ctx: Context) -> PrefixMessageEvent:
        number_of_slides = ev.number_of_slides | 10

        # set the googlesheet id and llm to ctx
        await ctx.set("llm", OpenAI(model="gpt-4o"))

        composio_toolset = ComposioToolSet()

        tools = composio_toolset.get_tools(actions=[
            Action.CODEINTERPRETER_EXECUTE_CODE,
            Action.CODEINTERPRETER_GET_FILE_CMD,
            Action.CODEINTERPRETER_RUN_TERMINAL_CMD,
            Action.GOOGLESHEETS_BATCH_GET
        ])
        await ctx.set("tools", tools)

        prefix_messages = [
            ChatMessage(
                role="system",
                content=(
                    f"""
                    You are an AI assistant specialized in creating PowerPoint presentations using the python-pptx library. 
                    Your task is to analyze the Google Sheets data from the provided spreadsheet ID: {ev.google_sheet_id}. 
                    Extract key insights and generate relevant charts based on this data. 
                    Finally, create a well-structured presentation that includes these charts and any necessary images, ensuring 
                    that the formatting is professional and visually appealing. Always create {number_of_slides} slides with more 
                    in-depth information covering all aspects of the sheet. When utilizing the Google Sheets tool, only the 
                    spreadsheet ID should be passed as input parameters.
                    NOTE: Mostly the user passes small sheets, so try to read the whole sheet at once and not via ranges.
                    """
                )
            )
        ]

        task = f"""
        Create a PowerPoint presentation from the Google Sheet: {ev.google_sheet_id}. 
        Create a sandbox First retrieve the sheets content, pip install python-pptx using the code interpreter, 
        and then use python-pptx. Then write code to create graphs from the data.
        Ensure the presentation is detailed, visually appealing, and contains {number_of_slides} slides. 
        Include charts and tables for key insights and ensure proper formatting.
        """

        print(task)

        await ctx.set("task", task)
        return PrefixMessageEvent(prefix_messages=prefix_messages)

    @step
    async def create_agent(self, ev: PrefixMessageEvent, ctx: Context) -> AgentEvent:
        tools = await ctx.get("tools")
        llm = await ctx.get("llm")
        prefix_messages = ev.prefix_messages

        agent = FunctionCallingAgentWorker(
            tools=tools,
            llm=llm,
            prefix_messages=prefix_messages,
            max_function_calls=15,
            allow_parallel_tool_calls=False,
            verbose=True
        ).as_agent()

        return AgentEvent(agent=agent)

    @step
    async def run_agent(self, ev: AgentEvent, ctx: Context) -> StopEvent:
        agent = ev.agent
        task = await ctx.get("task")
        response = agent.chat(task)
        return StopEvent(result=response)


def get_google_sheet_id_from_promot(user_prompt: str):
    system_prompt = """
    As a personal assistant, analyze the given prompt and extract the Google Sheet ID. A Google Sheet ID is a unique 
    string of characters that appears in the Google Sheet URL between /d/ and /edit.

    Return the information in the following JSON structure, maintaining the exact format:
    {
        "sheet_id": "string"
    }
    
    Note: A valid Google Sheet ID typically:
    - Contains letters, numbers, and special characters like '-' and '_'
    - Is approximately 44 characters long
    - Does not contain spaces or common URL characters like '/' or '?'
    
    Note: dont enclose in back tick like ```. just return plain JSON object.
    """

    # Send the preprocessed prompt to the LLM
    response = openai.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    print(response.choices[0].message.content)

    # Extract the response
    llm_response = json.loads(response.choices[0].message.content)
    print(llm_response)
    return llm_response


async def main():
    response = get_google_sheet_id_from_promot(
        user_prompt="create the presentation by refering to the data source at https://docs.google.com/spreadsheets/d/1JJZdYpyEFsF-IXUa5Ek30wlNdAueICMf26BLUQLnbuU/edit?gid=793403375#gid=793403375")
    w = PresentationGenerationWorkflow(timeout=100, verbose=True)
    result = await w.run(google_sheet_id=response['sheet_id'], number_of_slides=5)
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
