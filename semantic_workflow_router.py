from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.llms.ollama import OllamaLLM
from semantic_router.encoders import FastEmbedEncoder

calender_workflow = Route(
    name="calender_workflow",
    utterances=[
        "Can you schedule a meeting for tomorrow at 10 AM?",
        "Block my calendar for a team lunch next Friday at noon.",
        "Set up a meeting with the marketing team for Monday at 3 PM.",
        "I need to block my calendar for a doctor's appointment on Thursday at 5 PM.",
        "Schedule a brainstorming session for this Wednesday at 2 PM.",
        "Please block the whole day next Tuesday for a conference.",
        "Book a meeting with John and Sarah for 11 AM tomorrow.",
        "Can you arrange a meeting with the sales team on Friday at 4 PM?",
        "I need my calendar blocked for the next two hours starting now.",
        "Reserve 3 PM this Thursday for a project review meeting.",
        "Schedule an event on my calendar for Saturday evening at 7 PM.",
        "Please block time for a client call next Wednesday at 10 AM.",
        "I need to set up a meeting with the IT team this Friday at 11 AM.",
        "Mark my calendar as busy for lunch with the CEO at 1 PM tomorrow.",
        "Schedule a weekly stand-up meeting every Monday at 9 AM.",
        "Block my calendar for a webinar on artificial intelligence on Thursday at 6 PM.",
        "Reserve the entire afternoon next Wednesday for personal work.",
        "Can you set up a meeting with the HR team for next Monday at 10 AM?",
        "Block off 10 AM to 11:30 AM on Tuesday for a strategy session.",
        "Schedule a one-on-one meeting with my mentor for this Thursday at 4 PM.",
        "Mark 3 PM on Friday for a follow-up call with the client.",
        "Book a recurring meeting for the next four Wednesdays at 2 PM.",
        "Block my calendar for a family event on Sunday at 5 PM.",
        "Schedule a quick sync-up meeting tomorrow at 8 AM."
    ]
)

presentation_workflow = Route(
    name="presentation_workflow",
    utterances=[
        "Can you create a presentation using the sales data from the Google Sheet?",
        "Use the data in the Google Sheet to make a project update presentation.",
        "Please generate a presentation based on the budget report in the shared sheet.",
        "Create slides summarizing the marketing data in the Google Sheet.",
        "Make a presentation with the revenue figures from the spreadsheet.",
        "Can you prepare a PowerPoint using the product launch data from the sheet?",
        "Use the shared Google Sheet to create a summary slide deck.",
        "I need a presentation created using the quarterly performance data in the sheet.",
        "Take the data from the Google Sheet and make a client-ready presentation.",
        "Can you use the chart data from the sheet to build slides for tomorrow's meeting?",
        "Prepare a presentation from the inventory data in the spreadsheet.",
        "Create slides from the Google Sheet for the annual report.",
        "Make a pitch deck using the customer feedback data in the shared sheet.",
        "I need a summary presentation based on the Google Sheet's financial data.",
        "Please convert the data in the spreadsheet into a concise presentation.",
        "Build a presentation using the campaign performance metrics in the Google Sheet.",
        "Can you prepare a slide deck based on the attendance data in the shared sheet?",
        "Turn the Google Sheet data into a presentation for the board meeting.",
        "Use the figures in the shared sheet to design a detailed presentation.",
        "Make a presentation from the task completion data in the spreadsheet.",
        "Create a slide deck from the sales trends shown in the Google Sheet.",
        "Use the spreadsheet data to prepare a presentation for the weekly team meeting.",
        "Can you design slides from the survey results in the Google Sheet?",
        "Generate a presentation based on the revenue analysis data in the shared sheet.",
        "Make a presentation summarizing the milestones in the Google Sheet.",
        "I need slides prepared from the shared Google Sheet's project timeline.",
        "Build a presentation highlighting the top insights from the data in the spreadsheet."
    ]
)


def get_route(query: str):
    encoder = FastEmbedEncoder(name="snowflake/snowflake-arctic-embed-s")
    routes = [calender_workflow, presentation_workflow]
    llm = OllamaLLM(llm_name="qwen2.5:latest")

    sr = RouteLayer(encoder=encoder, routes=routes, llm=llm)
    return sr(query).name
