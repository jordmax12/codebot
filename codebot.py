import json
import os
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
import openai  # Install openai package: pip install openai
import requests

# Models
OLLAMA_MODEL = 'llama3.2:latest'
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.0-flash"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
GROK_MODEL = "grok-2-1212"

# Load environment variables from .env file
load_dotenv()

class OllamaClient:
    def __init__(self, model=OLLAMA_MODEL, host="http://localhost:11434"):
        self.model = model
        self.host = host
    
    def stream_query(self, prompt):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True
        }
        response = requests.post(f"{self.host}/api/generate", json=payload, stream=True)
        if response.status_code != 200:
            raise Exception(f"Ollama error: {response.text}")
        
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    yield data["response"]
                if data.get("done", False):
                    break

class GrokClient:
    def __init__(self, api_key, model=GROK_MODEL):
        self.client = openai.OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key
        )
        self.model = model
    
    def stream_query(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class OpenAIClient:
    def __init__(self, api_key, model=OPENAI_MODEL):
        self.client = openai.OpenAI(
            api_key=api_key
        )
        self.model = model
    
    def stream_query(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

class CodeBot:
    def __init__(self, index_file="index.json", root_dir="lambdas", model_type="ollama", ollama_host="http://localhost:11434"):
        self.root_dir = root_dir
        with open(index_file, "r") as f:
            self.index = json.load(f)
        
        # Get API keys from environment variables
        grok_api_key = os.getenv("X_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize the appropriate client based on model_type
        if model_type == "ollama":
            self.client = OllamaClient(host=ollama_host)
        elif model_type == "grok":
            if not grok_api_key:
                raise ValueError("Grok API key is required. Please set X_API_KEY in .env file.")
            self.client = GrokClient(grok_api_key)
        elif model_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in .env file.")
            self.client = OpenAIClient(openai_api_key)
        else:
            raise ValueError("Invalid model type. Choose 'ollama', 'grok', or 'openai'")
    
    def read_file(self, microservice, file_path):
        full_path = os.path.join(self.root_dir, microservice, file_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read()[:5000]  # Truncate to avoid overwhelming context
        return "File not found or inaccessible."
    
    def build_prompt(self, question, microservice=None, chat_history=None, file_content=None):
        index_str = json.dumps(self.index, indent=2)
        prompt = f"""
You are CodeBot, a helpful assistant for a serverless codebase. Here's the index:

{index_str}

The codebase is in `{self.root_dir}/`. Request file contents with "fetch <microservice>/<file_path>" (e.g., "fetch microservice-1/handler.js"). 
"""
        if microservice:
            microservice_data = self.index["microservices"][microservice]
            prompt += f"Focus on microservice: {microservice}\n"
            prompt += f"Microservice details: Functions - {microservice_data['serverless_functions']}, Helpers - {microservice_data['helpers']}, Controllers - {microservice_data['controllers']}\n"
        if chat_history:
            prompt += "Previous conversation context:\n"
            for msg in chat_history:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
        prompt += f"Answer: \"{question}\""
        if file_content:
            prompt += f"\nFile contents:\n```\n{file_content}\n```"
        return prompt
    
    def process_stream(self, question, microservice, history):
        """Stream response and handle file fetches, yielding chat history in messages format."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        history.append({"role": "user", "content": f"**You @ {timestamp}**: {question}"})
        yield history
        
        prompt = self.build_prompt(question, microservice, history)
        full_response = ""
        
        for chunk in self.client.stream_query(prompt):
            full_response += chunk
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = f"**CodeBot @ {timestamp}**: {full_response}"
            else:
                history.append({"role": "assistant", "content": f"**CodeBot @ {timestamp}**: {full_response}"})
            yield history
        
        if "fetch" in full_response.lower():
            for line in full_response.split("\n"):
                if line.startswith("fetch "):
                    try:
                        _, path = line.split("fetch ", 1)
                        microservice_to_use, file_path = path.strip().split("/", 1)
                        # Use the selected microservice or the one in the fetch path
                        microservice_to_use = microservice if microservice else microservice_to_use
                        file_content = self.read_file(microservice_to_use, file_path)
                        full_response += f"\nFetched {microservice_to_use}/{file_path}..."
                        if history[-1]["role"] == "assistant":
                            history[-1]["content"] = f"**CodeBot @ {timestamp}**: {full_response}"
                        else:
                            history.append({"role": "assistant", "content": f"**CodeBot @ {timestamp}**: {full_response}"})
                        yield history
                        
                        new_prompt = self.build_prompt(question, microservice, history, file_content)
                        for chunk in self.client.stream_query(new_prompt):
                            full_response += chunk
                            if history[-1]["role"] == "assistant":
                                history[-1]["content"] = f"**CodeBot @ {timestamp}**: {full_response}"
                            else:
                                history.append({"role": "assistant", "content": f"**CodeBot @ {timestamp}**: {full_response}"})
                            yield history
                        break
                    except ValueError:
                        full_response += "\nError: Invalid fetch format. Use 'fetch microservice/file'."
                        if history[-1]["role"] == "assistant":
                            history[-1]["content"] = f"**CodeBot @ {timestamp}**: {full_response}"
                        else:
                            history.append({"role": "assistant", "content": f"**CodeBot @ {timestamp}**: {full_response}"})
                        yield history

def create_interface():
    bot = CodeBot()
    microservices = sorted(list(bot.index["microservices"].keys()))
    
    css = """
    body { font-family: 'Arial', sans-serif; }
    .chatbox { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .message { padding: 10px; margin: 5px; border-radius: 5px; }
    .user-msg { background-color: #e3f2fd; color: #1976d2; }
    .bot-msg { background-color: #f5f5f5; color: #333; }
    .sidebar { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
    .footer { text-align: center; font-size: 12px; color: #777; }
    .dark body { background-color: #1a1a1a; color: #fff; }
    .dark .chatbox { background-color: #2d2d2d; box-shadow: 0 4px 8px rgba(0,0,0,0.3); }
    .dark .message { background-color: #3a3a3a; }
    .dark .user-msg { background-color: #2a5078; color: #e3f2fd; }
    .dark .bot-msg { background-color: #424242; color: #fff; }
    .dark .sidebar { background-color: #2d2d2d; }
    .dark .footer { color: #bbb; }
    """

    with gr.Blocks(title="CodeBot", css=css) as demo:
        gr.Markdown("# CodeBot\nYour Serverless Code Assistant", elem_classes=["header"])
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Microservices", elem_classes=["sidebar"])
                microservice_select = gr.Dropdown(choices=microservices, label="Select Microservice", value=microservices[0], filterable=True, allow_custom_value=False)
                clear_search = gr.Button("Clear", scale=1)
                model_select = gr.Dropdown(choices=["ollama", "grok", "openai"], label="Select Model", value="ollama")
                theme_toggle = gr.Button("Toggle Dark Mode")
                clear_button = gr.Button("Clear Chat")
            
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Conversation", type="messages", elem_classes=["chatbox"], height=500)
                with gr.Row():
                    question_input = gr.Textbox(placeholder="Ask about your codebase...", label="Your Question", scale=3)
                    send_button = gr.Button("Send", variant="primary", scale=1)
        
        gr.Markdown("Built with ❤️ by xAI | Feb 22, 2025", elem_classes=["footer"])

        chat_history = gr.State([])  # Persistent chat history
        selected_microservice = gr.State(microservices[0])  # Track selected microservice

        def clear_search_input():
            return microservices[0]

        def select_microservice(microservice, history):
            if microservice:
                timestamp = datetime.now().strftime("%H:%M:%S")
                history.append({"role": "assistant", "content": f"**CodeBot @ {timestamp}**: Added microservice: {microservice}"})
            return history, microservice

        def update_bot(model_type, history):
            return CodeBot(model_type=model_type), history

        def submit_question(question, history, microservice, model_type):
            if not question:
                return history, "", microservice, gr.update()
            bot, updated_history = update_bot(model_type, history)
            for new_history in bot.process_stream(question, microservice, updated_history):
                yield new_history, "", microservice, gr.update()
            return new_history, "", microservice, gr.update()

        microservice_select.change(
            fn=select_microservice,
            inputs=[microservice_select, chat_history],
            outputs=[chatbot, selected_microservice]
        )
        send_button.click(
            fn=submit_question,
            inputs=[question_input, chat_history, selected_microservice, model_select],
            outputs=[chatbot, question_input, selected_microservice, model_select]
        )
        question_input.submit(
            fn=submit_question,
            inputs=[question_input, chat_history, selected_microservice, model_select],
            outputs=[chatbot, question_input, selected_microservice, model_select]
        )
        clear_button.click(
            fn=lambda: ([], "", microservices[0], gr.update()),  # Reset to first microservice on clear
            outputs=[chat_history, question_input, selected_microservice, model_select]
        )
        clear_search.click(
            fn=clear_search_input,
            inputs=[],
            outputs=[microservice_select]
        )
        model_select.change(
            fn=update_bot,
            inputs=[model_select, chat_history],
            outputs=[gr.State(), chat_history]  # Return new bot and preserved history
        ).then(
            fn=lambda history: (history, gr.update()),
            inputs=[chat_history],
            outputs=[chatbot, model_select]
        )
        theme_toggle.click(
            fn=lambda x: not x,
            inputs=[gr.State(False)],
            outputs=[gr.State(True)]
        ).then(
            fn=lambda history, dark: history,
            inputs=[chat_history, gr.State(True)],
            outputs=[chatbot]
        )

    return demo

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(inbrowser=True)
