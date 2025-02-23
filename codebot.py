import json
import os
import gradio as gr
import openai
import requests
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
import tiktoken  # Install tiktoken: pip install tiktoken

# Models
OLLAMA_MODEL = 'llama3.2'
OLLAMA_HOST = "http://localhost:11434/v1"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_HOST = "https://api.openai.com/v1"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_HOST = "https://generativelanguage.googleapis.com/v1beta/"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
GROK_MODEL = "grok-2-1212"
GROK_HOST = "https://api.x.ai/v1"

# Load environment variables from .env file
load_dotenv()

class Client:
    def __init__(self, api_key, model=OLLAMA_MODEL, host=OLLAMA_HOST):
        print(f"Initializing client with host: {host}, api_key: {api_key}, model: {model}")
        self.client = openai.OpenAI(
            base_url=host,
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

class AnthropicClient:
    def __init__(self, api_key, model=ANTHROPIC_MODEL):
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def stream_query(self, prompt):
        with self.client.messages.stream(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024  # Optional: match your example's max_tokens or adjust as needed
        ) as stream:
            for text in stream.text_stream:
                yield text

class CodeBot:
    def __init__(self, index_file="index.json", root_dir="lambdas", model_type="ollama", ollama_host="http://localhost:11434"):
        self.root_dir = root_dir
        with open(index_file, "r") as f:
            self.index = json.load(f)
        
        # Get API keys from environment variables
        grok_api_key = os.getenv("X_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        # Initialize the appropriate client based on model_type
        if model_type == "ollama":
            self.client = Client(api_key="ollama", host=OLLAMA_HOST)
        elif model_type == "grok":
            if not grok_api_key:
                raise ValueError("Grok API key is required. Please set X_API_KEY in .env file.")
            self.client = Client(api_key=grok_api_key, model=GROK_MODEL, host=GROK_HOST)
        elif model_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY in .env file.")
            self.client = Client(api_key=openai_api_key, model=OPENAI_MODEL, host=OPENAI_HOST)
        elif model_type == "gemini":
            if not gemini_api_key:
                raise ValueError("Gemini API key is required. Please set GEMINI_API_KEY in .env file.")
            self.client = Client(api_key=gemini_api_key, model=GEMINI_MODEL, host=GEMINI_HOST)
        elif model_type == "anthropic":
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required. Please set ANTHROPIC_API_KEY in .env file.")
            self.client = AnthropicClient(anthropic_api_key)
        else:
            raise ValueError("Invalid model type. Choose 'ollama', 'grok', 'openai', 'gemini', or 'anthropic'")
        self.model_type = model_type  # Track the model type for display
    
    def read_file(self, microservice, file_path):
        full_path = os.path.join(self.root_dir, microservice, file_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read()[:5000]  # Truncate to avoid overwhelming context
        return "File not found or inaccessible."
    
    def estimate_tokens(self, text):
        """Estimate token count for a given text based on model type."""
        if self.model_type in ["openai", "grok"]:
            try:
                encoding = tiktoken.encoding_for_model(self.model_type == "openai" and OPENAI_MODEL or GROK_MODEL)
                return len(encoding.encode(text))
            except Exception as e:
                print(f"Token estimation error for {self.model_type}: {e}")
                # Fallback: approximate 4 chars per token
                return len(text) // 4
        else:  # Ollama, Anthropic, Gemini
            # Rough approximation: 4 characters ≈ 1 token
            return len(text) // 4
    
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
        """Stream response and handle file fetches, yielding chat history and token updates separately."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        history.append({"role": "user", "content": f"**You @ {timestamp}**: {question}"})
        yield history, gr.update(value="Calculating...")  # Start with "Calculating..."
        
        prompt = self.build_prompt(question, microservice, history)
        full_response = ""
        prompt_tokens = self.estimate_tokens(prompt)
        
        for chunk in self.client.stream_query(prompt):
            full_response += chunk
            model_display = f"({self.model_type.capitalize()})"
            if history and history[-1]["role"] == "assistant":
                history[-1]["content"] = f"**CodeBot {model_display} @ {timestamp}**: {full_response}"
            else:
                history.append({"role": "assistant", "content": f"**CodeBot {model_display} @ {timestamp}**: {full_response}"})
            yield history, gr.update(value="Calculating...")  # Show "Calculating..." during streaming
        
        response_tokens = self.estimate_tokens(full_response)
        total_tokens = prompt_tokens + response_tokens
        
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
                            history[-1]["content"] = f"**CodeBot {model_display} @ {timestamp}**: {full_response}"
                        else:
                            history.append({"role": "assistant", "content": f"**CodeBot {model_display} @ {timestamp}**: {full_response}"})
                        yield history, gr.update(value="Calculating...")  # Show "Calculating..." during fetch streaming
                        
                        new_prompt = self.build_prompt(question, microservice, history, file_content)
                        new_prompt_tokens = self.estimate_tokens(new_prompt)
                        for chunk in self.client.stream_query(new_prompt):
                            full_response += chunk
                            if history[-1]["role"] == "assistant":
                                history[-1]["content"] = f"**CodeBot {model_display} @ {timestamp}**: {full_response}"
                            else:
                                history.append({"role": "assistant", "content": f"**CodeBot {model_display} @ {timestamp}**: {full_response}"})
                            yield history, gr.update(value="Calculating...")  # Show "Calculating..." during fetch response
                        response_tokens += self.estimate_tokens(full_response[len(full_response) - len(chunk):])
                        total_tokens = prompt_tokens + new_prompt_tokens + response_tokens
                        yield history, total_tokens  # Final yield with total tokens
                        break
                    except ValueError:
                        full_response += "\nError: Invalid fetch format. Use 'fetch microservice/file'."
                        if history[-1]["role"] == "assistant":
                            history[-1]["content"] = f"**CodeBot {model_display} @ {timestamp}**: {full_response}"
                        else:
                            history.append({"role": "assistant", "content": f"**CodeBot {model_display} @ {timestamp}**: {full_response}"})
                        yield history, 0  # Yield 0 tokens on error
        else:
            yield history, total_tokens  # Final yield with total tokens if no fetch

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
    .model-indicator { color: #2ecc71; font-weight: bold; } /* Green for LLM indicator, like Git branch */
    .token-count { font-size: 14px; color: #666; margin-top: 5px; }
    """

    with gr.Blocks(title="CodeBot", css=css) as app:
        gr.Markdown("# CodeBot\nYour Codebase Assistant", elem_classes=["header"])
        
        with gr.Row():
            with gr.Column(scale=1):
                microservice_select = gr.Dropdown(choices=microservices, label="Select Microservice", value=microservices[0], filterable=True, allow_custom_value=False)
                model_select = gr.Dropdown(choices=["ollama", "grok", "openai", "gemini", "anthropic"], label="Select Model", value="ollama")
                clear_button = gr.Button("Clear Chat")
                token_count = gr.Number(label="Tokens Used (Last Request)", value=0, elem_classes=["token-count"])
            
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(label="Conversation", type="messages", elem_classes=["chatbox"], height=500)
                with gr.Row():
                    question_input = gr.Textbox(placeholder="Ask about your codebase...", label="Your Question", scale=3)
                    send_button = gr.Button("Send", variant="primary", scale=1)
        
        gr.Markdown("Built by Jordan", elem_classes=["footer"])

        chat_history = gr.State([])  # Persistent chat history
        selected_microservice = gr.State(microservices[0])  # Track selected microservice
        token_count_state = gr.State(0)  # Track token count

        def select_microservice(microservice, history):
            if microservice:
                timestamp = datetime.now().strftime("%H:%M:%S")
                history.append({"role": "assistant", "content": f"**CodeBot @ {timestamp}**: Added microservice: {microservice}"})
            return history, microservice, gr.update()

        def update_bot(model_type, history):
            return CodeBot(model_type=model_type), history

        def submit_question(question, history, microservice, model_type):
            if not question:
                return history, "", microservice, gr.update(), gr.update(value=0)  # Reset token count to 0
            bot, updated_history = update_bot(model_type, history)
            for new_history, tokens in bot.process_stream(question, microservice, updated_history):
                if isinstance(tokens, int):  # Final token count
                    yield new_history, "", microservice, gr.update(), gr.update(value=tokens)
                else:  # Streaming phase
                    yield new_history, "", microservice, gr.update(), gr.update(value=0)  # Show 0 during calculation
            if isinstance(tokens, int):  # Final return
                return new_history, "", microservice, gr.update(), gr.update(value=tokens)
            return new_history, "", microservice, gr.update(value=0), gr.update(value=tokens)

        microservice_select.change(
            fn=select_microservice,
            inputs=[microservice_select, chat_history],
            outputs=[chatbot, selected_microservice, token_count]
        )
        send_button.click(
            fn=submit_question,
            inputs=[question_input, chat_history, selected_microservice, model_select],
            outputs=[chatbot, question_input, selected_microservice, model_select, token_count]
        )
        question_input.submit(
            fn=submit_question,
            inputs=[question_input, chat_history, selected_microservice, model_select],
            outputs=[chatbot, question_input, selected_microservice, model_select, token_count]
        )
        clear_button.click(
            fn=lambda: ([], "", microservices[0], gr.update(), 0),  # Reset chat, microservice, and token count to 0
            outputs=[chat_history, question_input, selected_microservice, model_select, token_count]
        )
        model_select.change(
            fn=update_bot,
            inputs=[model_select, chat_history],
            outputs=[gr.State(), chat_history]  # Return new bot and preserved history
        ).then(
            fn=lambda history: (history, gr.update(), gr.update()),
            inputs=[chat_history],
            outputs=[chatbot, model_select, token_count]
        )

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(inbrowser=True)