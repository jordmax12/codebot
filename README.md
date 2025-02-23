# CodeBot

A local chatbot for understanding a serverless codebase.

## Setup
1. Activate the Anaconda environment: `conda activate codebot`
2. Install dependencies: `conda install pyyaml requests`
3. Install additional dependencies: `pip install gradio openai python-dotenv anthropic tiktoken`
4. Run Ollama in a separate terminal: `ollama run codellama`
5. Place this app in the parent directory of `lambdas/`.

## Usage
1. Build the index: `python indexer.py`
2. Launch the UI: `python codebot.py`
3. Open the browser link (e.g., http://127.0.0.1:7860) and ask questions!

## Features
- Real-time streaming responses.
- Chat history with timestamps.
- Microservice selector sidebar.
- Dark mode toggle.
- Clear chat button.

## Notes
- Supports Node.js primarily; extend `indexer.py` for Python/Go if needed.
- Index stored in `index.json`.

