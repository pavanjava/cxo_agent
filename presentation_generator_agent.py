import streamlit as st
from composio_llamaindex import ComposioToolSet, App, Action
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import shutil
import os
import glob


class PowerPointGenerator:
    def __init__(self, google_sheet_id, model: str = 'gpt-4o'):
        self.google_sheet_id = google_sheet_id
        self.llm = OpenAI(model=model, timeout=300.0)
        Settings.llm = self.llm
        self.composio_toolset = ComposioToolSet()
        self.tools = self.composio_toolset.get_tools(actions=[
            Action.CODEINTERPRETER_EXECUTE_CODE,
            Action.CODEINTERPRETER_GET_FILE_CMD,
            Action.CODEINTERPRETER_RUN_TERMINAL_CMD,
            Action.GOOGLESHEETS_BATCH_GET
        ])
        self.agent = self.create_agent()

    def create_agent(self):
        prefix_messages = [
            ChatMessage(
                role="system",
                content=(
                    f"""
                    You are an AI assistant specialized in creating PowerPoint presentations using the python-pptx library. 
                    Your task is to analyze the Google Sheets data from the provided spreadsheet ID: {self.google_sheet_id}. 
                    Extract key insights and generate relevant charts based on this data. 
                    Finally, create a well-structured presentation that includes these charts and any necessary images, ensuring 
                    that the formatting is professional and visually appealing. Always create 10 slides with more 
                    in-depth information covering all aspects of the sheet. When utilizing the Google Sheets tool, only the 
                    spreadsheet ID should be passed as input parameters.
                    NOTE: Mostly the user passes small sheets, so try to read the whole sheet at once and not via ranges.
                    """
                )
            )
        ]

        return FunctionCallingAgentWorker(
            tools=self.tools,
            llm=self.llm,
            prefix_messages=prefix_messages,
            max_function_calls=15,
            allow_parallel_tool_calls=False,
            verbose=True
        ).as_agent()

    @staticmethod
    def copy_pptx_to_current_directory():
        source_dir = '/Users/pavanmantha/.composio/output/'
        destination_dir = os.getcwd()
        pptx_files = glob.glob(os.path.join(source_dir, '*.pptx'))

        for file_path in pptx_files:
            filename = os.path.basename(file_path)
            destination_path = os.path.join(destination_dir, filename)
            shutil.move(file_path, destination_path)
            return destination_path  # Return the path of the moved file

        print("All PowerPoint files have been moved.")
        return None

    def generate_presentation(self, number_of_slides: int = 10):
        task = f"""
        Create a PowerPoint presentation from the Google Sheet: {self.google_sheet_id}. 
        Create a sandbox First retrieve the sheets content, pip install python-pptx using the code interpreter, 
        and then use python-pptx. Then write code to create graphs from the data.
        Ensure the presentation is detailed, visually appealing, and contains {number_of_slides} slides. 
        Include charts and tables for key insights and ensure proper formatting.
        """

        self.agent.chat(task)
        return self.copy_pptx_to_current_directory()


def main():
    st.set_page_config(
        page_title="PowerPoint Agent",
        page_icon="📊",
        layout="wide"
    )

    # Sidebar with How to Use instructions
    with st.sidebar:
        st.title("How to Use")
        st.markdown("""
        ### Steps to Generate Your Presentation
        
        1. **Find your Google Sheet ID**
           - Open your Google Sheet
           - Copy the ID from the URL
           - Example ID format: `1JJZdYpyEFsF-IXUa5Ek30wlNdAueICMf26BLUQLnbuU`
        
        2. **Configure Settings**
           - Paste the Sheet ID in the input field
           - Select your preferred GPT model
           - GPT-4: More capable but slower
           - GPT-3.5: Faster but less sophisticated
        
        3. **Generate & Download**
           - Click "Generate Presentation"
           - Wait for processing (typically 2-5 minutes)
           - Download the generated PPTX file
        
        ### Important Notes
        - Ensure your Google Sheet is accessible
        - Keep your sheet structure clean and organized
        - Larger sheets may take longer to process
        
        ### Need Help?
        If you encounter any issues, check:
        - Sheet permissions
        - Internet connection
        - API key configuration
        """)

    # Main content
    st.title("PowerPoint Generator Agent")
    st.markdown("""
    Transform your Google Sheets data into professional PowerPoint presentations using AI.
    """)

    # Load environment variables
    load_dotenv()

    # Create two columns for inputs
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Input for Google Sheet ID
        google_sheet_id = st.text_input(
            "Google Sheet ID",
            placeholder="Enter your Google Sheet ID...",
            help="You can find the Sheet ID in the URL of your Google Sheet"
        )

    with col2:
        # Input for number of sheets for presentation
        no_of_sheets = st.text_input(
            "number of slides",
            placeholder="Enter your count for sliders",
            help="input your count for sliders that you need the agent to generate"
        )

    with col3:
        # Model selection
        model = st.selectbox(
            "Select GPT Model",
            options=['gpt-4', 'gpt-4o', 'gpt-3.5-turbo'],
            help="Choose the GPT model to use for presentation generation"
        )

    # Generate button
    if st.button("Generate Presentation", type="primary", use_container_width=True):
        if not google_sheet_id:
            st.error("Please enter a Google Sheet ID")
            return

        try:
            with st.spinner("Generating your presentation... This may take a few minutes."):
                # Create generator instance
                ppt_generator = PowerPointGenerator(google_sheet_id, model=model)

                # Generate presentation and get the file path
                presentation_path = ppt_generator.generate_presentation(number_of_slides=no_of_sheets)

                if presentation_path and os.path.exists(presentation_path):
                    # Read the file for download
                    with open(presentation_path, "rb") as file:
                        presentation_data = file.read()

                    # Success message and download button in columns
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.success("Generation complete!")
                    with col2:
                        st.download_button(
                            label="Download Presentation",
                            data=presentation_data,
                            file_name="generated_presentation.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            use_container_width=True
                        )
                else:
                    st.error("Failed to generate presentation. Please try again.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
