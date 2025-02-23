import json
import os
import gradio as gr
import openai
import requests
import tiktoken
import faiss
import numpy as np
import re
from anthropic import Anthropic
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer 

# Models (unchanged)
OLLAMA_MODEL = 'llama3.2'
OLLAMA_HOST = "http://localhost:11434/v1"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_HOST = "https://api.openai.com/v1"
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_HOST = "https://generativelanguage.googleapis.com/v1beta/"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"
GROK_MODEL = "grok-2-1212"
GROK_HOST = "https://api.x.ai/v1"

load_dotenv()

# Client classes (unchanged)
class Client:
    def __init__(self, api_key, model=OLLAMA_MODEL, host=OLLAMA_HOST):
        self.client = openai.OpenAI(base_url=host, api_key=api_key)
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
            max_tokens=1024
        ) as stream:
            for text in stream.text_stream:
                yield text

class CodeBot:
    def __init__(self, index_file="index.json", root_dir="lambdas", model_type="ollama", ollama_host="http://localhost:11434"):
        self.root_dir = root_dir
        with open(index_file, "r") as f:
            self.index = json.load(f)
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store, self.file_map = self.build_vector_store()
        
        grok_api_key = os.getenv("X_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        gemini_api_key = os.getenv("GEMINI_API_KEY")

        if model_type == "ollama":
            self.client = Client(api_key="ollama", host=OLLAMA_HOST)
        elif model_type == "grok":
            if not grok_api_key:
                raise ValueError("Grok API key is required.")
            self.client = Client(api_key=grok_api_key, model=GROK_MODEL, host=GROK_HOST)
        elif model_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required.")
            self.client = Client(api_key=openai_api_key, model=OPENAI_MODEL, host=OPENAI_HOST)
        elif model_type == "gemini":
            if not gemini_api_key:
                raise ValueError("Gemini API key is required.")
            self.client = Client(api_key=gemini_api_key, model=GEMINI_MODEL, host=GEMINI_HOST)
        elif model_type == "anthropic":
            if not anthropic_api_key:
                raise ValueError("Anthropic API key is required.")
            self.client = AnthropicClient(anthropic_api_key)
        else:
            raise ValueError("Invalid model type.")
        self.model_type = model_type
    
    def minify_code(self, content):
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        content = re.sub(r'\s+', ' ', content).strip()
        return content[:2000]
    
    def build_vector_store(self):
        file_map = {}
        embeddings = []
        idx = 0
        for microservice, data in self.index["microservices"].items():
            for file_type in ["serverless_functions", "helpers", "controllers"]:
                for file_entry in data.get(file_type, []):
                    if isinstance(file_entry, dict) and "file" in file_entry:
                        file_path = file_entry["file"]
                    else:
                        continue
                    
                    if not file_path or file_path.startswith(".") and len(file_path) <= 3:
                        continue
                    
                    full_path = os.path.join(self.root_dir, microservice, file_path)
                    if os.path.exists(full_path):
                        try:
                            with open(full_path, "r") as f:
                                content = f.read()
                            minified = self.minify_code(content)
                            embedding = self.embedder.encode(minified, convert_to_numpy=True)
                            embeddings.append(embedding)
                            file_map[idx] = (microservice, file_path, minified)
                            idx += 1
                        except Exception as e:
                            print(f"Error processing {full_path}: {e}")
        
        if not embeddings:
            dimension = self.embedder.get_sentence_embedding_dimension()
            index = faiss.IndexFlatL2(dimension)
            return index, file_map
        
        dimension = embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        return index, file_map
    
    def read_file(self, microservice, file_path):
        full_path = os.path.join(self.root_dir, microservice, file_path)
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return self.minify_code(f.read())[:2000]
        return "File not found."
    
    def estimate_tokens(self, text):
        if self.model_type in ["openai", "grok"]:
            try:
                encoding = tiktoken.encoding_for_model(self.model_type == "openai" and OPENAI_MODEL or GROK_MODEL)
                return len(encoding.encode(text))
            except:
                return len(text) // 4
        return len(text) // 4
    
    def retrieve_relevant_files(self, question, k=5):
        question_embedding = self.embedder.encode(question, convert_to_numpy=True)
        distances, indices = self.vector_store.search(np.array([question_embedding]), k)
        retrieved = []
        for idx in indices[0]:
            if idx in self.file_map:
                microservice, file_path, content = self.file_map[idx]
                retrieved.append((microservice, file_path, content))
        return retrieved
    
    def summarize_index(self):
        summary = []
        for microservice, data in self.index["microservices"].items():
            files = []
            for file_type in ["serverless_functions", "helpers", "controllers"]:
                for entry in data.get(file_type, []):
                    if "file" in entry:
                        files.append(f"{file_type[0]}:{entry['file']}")
            summary.append(f"{microservice}:[{','.join(files)}]")
        return ";".join(summary)
    
    def build_prompt(self, question, chat_history=None, retrieved_files=None):
        index_summary = self.summarize_index()
        prompt = f"CodeBot: Codebase in `{self.root_dir}`. Index: {index_summary}. Q: {question}"
        if chat_history:
            prompt += "\nHist:"
            for msg in chat_history[-3:]:
                prompt += f"{msg['role'][0]}:{msg['content'][:100]}|"
        if retrieved_files:
            prompt += "\nFiles:"
            for microservice, file_path, content in retrieved_files:
                prompt += f"{microservice}/{file_path}:{content[:500]};"
        prompt += "\nMore files? Suggest fetch <microservice>/<file>. Answer fully if possible."
        return prompt
    
    def process_stream(self, question, microservice, history, max_iterations=3):
        timestamp = datetime.now().strftime("%H:%M:%S")
        history.append({"role": "user", "content": f"**You @ {timestamp}**: {question}"})
        yield history, gr.update(value="Calculating...")
        
        retrieved_files = self.retrieve_relevant_files(question) if not microservice else []
        if microservice:  # If microservice selected, prioritize its files
            for file_type in ["serverless_functions", "helpers", "controllers"]:
                for entry in self.index["microservices"][microservice].get(file_type, []):
                    if "file" in entry:
                        content = self.read_file(microservice, entry["file"])
                        retrieved_files.append((microservice, entry["file"], content))
        
        iteration = 0
        full_response = ""
        total_tokens = 0
        
        while iteration < max_iterations:
            prompt = self.build_prompt(question, history, retrieved_files)
            prompt_tokens = self.estimate_tokens(prompt)
            response = ""
            
            for chunk in self.client.stream_query(prompt):
                response += chunk
                model_display = f"({self.model_type.capitalize()})"
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] = f"**CodeBot {model_display} @ {timestamp}**: {full_response + response}"
                else:
                    history.append({"role": "assistant", "content": f"**CodeBot {model_display} @ {timestamp}**: {response}"})
                yield history, gr.update(value="Calculating...")
            
            full_response += response
            response_tokens = self.estimate_tokens(response)
            total_tokens += prompt_tokens + response_tokens
            
            # Check for fetch suggestions and process them
            fetch_pattern = re.compile(r"fetch\s+([^\s/]+)/([^\s]+)")
            fetches = fetch_pattern.findall(response)
            if fetches and iteration < max_iterations - 1:
                for microservice_to_fetch, file_path in fetches:
                    if (microservice_to_fetch, file_path) not in [(m, f) for m, f, _ in retrieved_files]:
                        file_content = self.read_file(microservice_to_fetch, file_path)
                        if file_content != "File not found.":
                            retrieved_files.append((microservice_to_fetch, file_path, file_content))
                            full_response += f"\nFetched {microservice_to_fetch}/{file_path}"
                            history[-1]["content"] = f"**CodeBot {model_display} @ {timestamp}**: {full_response}"
                            yield history, gr.update(value="Calculating...")
                iteration += 1
            else:
                break  # Exit if no fetches or max iterations reached
        
        yield history, total_tokens

# Interface (unchanged except process_stream signature)
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
        .model-indicator { color: #2ecc71; font-weight: bold; }
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

        chat_history = gr.State([])
        selected_microservice = gr.State(microservices[0])
        token_count_state = gr.State(0)

        def select_microservice(microservice, history):
            if microservice:
                timestamp = datetime.now().strftime("%H:%M:%S")
                history.append({"role": "assistant", "content": f"**CodeBot @ {timestamp}**: Added microservice: {microservice}"})
            return history, microservice, gr.update()

        def update_bot(model_type, history):
            return CodeBot(model_type=model_type), history

        def submit_question(question, history, microservice, model_type):
            if not question:
                return history, "", microservice, gr.update(), gr.update(value=0)
            bot, updated_history = update_bot(model_type, history)
            for new_history, tokens in bot.process_stream(question, microservice, updated_history):
                if isinstance(tokens, int):
                    yield new_history, "", microservice, gr.update(), gr.update(value=tokens)
                else:
                    yield new_history, "", microservice, gr.update(), gr.update(value=0)
            if isinstance(tokens, int):
                return new_history, "", microservice, gr.update(), gr.update(value=tokens)
            return new_history, "", microservice, gr.update(value=0), gr.update(value=tokens)

        microservice_select.change(fn=select_microservice, inputs=[microservice_select, chat_history], outputs=[chatbot, selected_microservice, token_count])
        send_button.click(fn=submit_question, inputs=[question_input, chat_history, selected_microservice, model_select], outputs=[chatbot, question_input, selected_microservice, model_select, token_count])
        question_input.submit(fn=submit_question, inputs=[question_input, chat_history, selected_microservice, model_select], outputs=[chatbot, question_input, selected_microservice, model_select, token_count])
        clear_button.click(fn=lambda: ([], "", microservices[0], gr.update(), 0), outputs=[chat_history, question_input, selected_microservice, model_select, token_count])
        model_select.change(fn=update_bot, inputs=[model_select, chat_history], outputs=[gr.State(), chat_history]).then(
            fn=lambda history: (history, gr.update(), gr.update()), inputs=[chat_history], outputs=[chatbot, model_select, token_count]
        )

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(inbrowser=True)
