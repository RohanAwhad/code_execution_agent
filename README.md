Chatbot with Code Execution and Web Search Integration

This repository provides a Streamlit-based chatbot that integrates AI-powered chat functionality, Python code execution in a Jupyter notebook environment, and web search capabilities via the Brave Search API. The bot allows users to interact in a conversational format, upload various file types, execute Python code, and retrieve web search results. The key components include OpenAI’s GPT models, Jupyter for live code execution, and Streamlit for the frontend interface.

Features

1.	AI-Powered Chat Interface: Users can converse with the chatbot, ask questions, and receive responses from GPT models powered by OpenAI.
2.	Python Code Execution: Users can input Python code directly in the chat, which is executed in a Jupyter notebook backend, and the results or generated plots are returned within the chat interface.
3.	File Upload Support:
	•	Images (.png, .jpg, .jpeg)
	•	PDFs
	•	Spreadsheets (.csv, .xls, .xlsx)
	•	Text files (.txt)
	Uploaded files are processed based on their type and displayed or stored for further use.
4.	Web Search via Brave Search API: Users can perform web searches by entering a query, and the results are fetched using Brave Search, which are then presented in the chat.
5.	Persistent Chat History: Chat history is saved locally, allowing users to access and review previous conversations, including code execution outputs.

Installation

Prerequisites

Make sure you have the following installed on your system:

	•	Python 3.8+
	•	Docker (for running in a container)
	•	OpenAI API Key
	•	Brave Search API Key

Local Setup

	1.	Clone the Repository:

git clone https://github.com/your-username/chatbot-with-code-execution.git
cd chatbot-with-code-execution

	2.	Create a Virtual Environment and Install Dependencies:

Create a virtual environment and install the required Python packages:

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

	3.	Set Up Environment Variables:

Create a .env file in the root of the project and include your API keys:

OPENAI_API_KEY=your_openai_api_key
BRAVE_SEARCH_AI_API_KEY=your_brave_search_api_key

	4.	Run the Streamlit App:

streamlit run streamlit_chatbot.py

After running this command, the app will be accessible locally at http://localhost:8501.

Docker Setup

If you prefer to run the app in a Docker container, follow these steps.

	1.	Make the Run Script Executable:

The repository includes a run.sh file that sets up the necessary environment and runs the app in Docker. You need to make it executable:

chmod +x run.sh

	2.	Run the Application:

Now, execute the run.sh script to build and run the Docker container:

./run.sh

This script will:

	•	Create the necessary data and history directories for storing files and chat history.
	•	Build the Docker image and run it, exposing port 8501 for the Streamlit interface.
	•	Pass the OpenAI API key as an environment variable to the container.

Once the container is running, you can access the application at http://localhost:8501.

Script Breakdown

Here is what the run.sh script does:

#!/bin/zsh

if [ ! -d "data" ]; then
    mkdir data
fi
if [ ! -d "history" ]; then
    mkdir history
fi
docker build -t code_execution_agent . && docker run -p8501:8501 --rm -e OPENAI_API_KEY=$OPENAI_API_KEY -v "$(pwd)/data:/app/data" -v "$(pwd)/history:/app/history" code_execution_agent

	•	Creates Directories: Checks if the data and history directories exist, and if not, it creates them.
	•	Builds Docker Image: Builds a Docker image named code_execution_agent from the current directory.
	•	Runs the Docker Container: The container is run with the following options:
	•	Exposes port 8501 for the Streamlit app.
	•	Passes the OpenAI API key as an environment variable.
	•	Mounts local data and history directories into the container for persistent file and chat history storage.
	•	Automatically removes the container when stopped.

Usage

Chat Interface

	•	Interact with the AI: Type a message in the input box, and the chatbot will respond using GPT. The AI can answer general questions, provide explanations, or assist with coding tasks.

Code Execution

	•	Run Python Code: You can input Python code directly in the chat interface. The code will be executed in a live Jupyter notebook kernel, and the output will be displayed in the chat. This includes any printed results, plots, or errors.

File Uploads

	•	Upload Files: The chatbot supports the upload of images, PDFs, spreadsheets, and text files. It processes and displays the contents of the files or stores them for further use.

Web Search

	•	Search the Web: Users can ask the bot to search for information online. Using the Brave Search API, the chatbot will retrieve and display search results in the chat.

Persistent Conversations

	•	Chat History: Conversations are saved in the history directory. You can revisit old chats by clicking on them in the sidebar.

Project Structure

	•	streamlit_chatbot.py: The main application file that sets up the Streamlit interface, handles user interactions, file uploads, and code execution.
	•	requirements.txt: Lists the Python dependencies for the project.
	•	run.sh: Shell script to build and run the Docker container.
	•	history/: Directory where chat history is saved.
	•	data/: Directory where uploaded files are temporarily stored.

Contributing

We welcome contributions to improve this project. If you’d like to contribute, follow these steps:

	1.	Fork the repository.
	2.	Create a new branch for your feature or bug fix.
	3.	Make your changes and commit them.
	4.	Submit a pull request with a detailed description of the changes.

License

This project is licensed under the MIT License.

Enjoy building with this interactive chatbot and code execution agent! Feel free to suggest improvements or report any issues.
