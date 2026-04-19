import os
import sys
import json
import time
import subprocess
import readline
import tiktoken
from datetime import datetime
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# RAG Dependencies
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

class Config:
    def __init__(self, path='config.json'):
        self.path = path
        self.data = {
            'token_threshold': 5000,
            'url': 'http://localhost:1234/v1',
            'port': 1234,
            'temperature': 0.7,
            'max_tokens': 8192,
            'model': 'local-model'
        }
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                try:
                    loaded = json.load(f)
                    self.data.update(loaded)
                except json.JSONDecodeError:
                    pass
        else:
            self.save()

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save()

class MemoryManager:
    def __init__(self):
        if not HAS_CHROMA:
            print("Warning: ChromaDB not installed. RAG features disabled.")
            return
        try:
            self.client = chromadb.PersistentClient(path="./rag_memory")
            self.collection = self.client.get_or_create_collection(name="user_facts")
        except Exception as e:
            print(f"Error initializing RAG: {e}")
            self.client = None

    def save_memory(self, content):
        if not self.client:
            return "RAG not available."
        try:
            # Simple embedding using a dummy method or relying on Chroma's default if installed with transformers
            # For this script, we assume chromadb has a default embedding function or we use a placeholder
            # In a real env, one might need to install sentence-transformers.
            # Here we rely on Chroma's default behavior.
            self.collection.add(
                documents=[content],
                ids=[str(int(time.time()))]
            )
            return "Memory saved."
        except Exception as e:
            return f"Error saving memory: {e}"

    def load_memory(self, query):
        if not self.client:
            return "RAG not available."
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=5
            )
            return "\n".join(results['documents'][0]) if results['documents'][0] else "No memories found."
        except Exception as e:
            return f"Error loading memory: {e}"

    def load_all_memory(self):
        if not self.client:
            return "RAG not available."
        try:
            results = self.collection.get()
            return "\n".join(results['documents']) if results['documents'] else "No memories found."
        except Exception as e:
            return f"Error loading all memory: {e}"

class ChatInterface:
    def __init__(self):
        self.config = Config()
        self.memory = MemoryManager()
        self.client = OpenAI(
            base_url=self.config.get('url', 'http://localhost:1234/v1'),
            api_key="lm-studio" # LMStudio often uses a dummy key or empty
        )
        self.history_file = "chat_history.json"
        self.history = []
        self.messages = [] # Raw messages for API
        self.token_count = 0
        self.load_history()
        self.setup_readline()

    def load_history(self):
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                try:
                    self.history = json.load(f)
                    # Reconstruct messages list for API usage
                    self.messages = [msg for msg in self.history if msg.get('role') in ['system', 'user', 'assistant']]
                except json.JSONDecodeError:
                    self.history = []
                    self.messages = []
        else:
            self.history = []
            self.messages = []

    def save_history(self):
        # Save the current history state
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def setup_readline(self):
        completer = readline.get_completer()
        readline.set_completer(self.complete)
        readline.parse_and_bind("tab: complete")

    def complete(self, text, state):
        if text.startswith("/"):
            commands = ["exit", "clear", "history", "memory", "messages", "shrink", "threshold", "tools", "tokens", "config", "url", "help"]
            return [cmd for cmd in commands if cmd.startswith(text[1:])][state]
        return None

    def get_system_prompt(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prompt = f'''You are a professional system assistant.
"You can read/write files, run bash commands, and manage a long-term memory. "
"If the user asks to see or load all memories, use 'load_all_memory'. "
"If the user tells you a fact about themselves or the system, use 'save_memory' to remember it. "
"If you need to recall a known fact, use 'get_memory'. "
"Don't just make things up. "
"Use the available tools if they help you solve a problem. "
"Work efficiently by chaining commands together. "
"Always provide clear and professional answers. "
"Load all memory entries if you're missing information. "
Current Date & Time: {now}'''
        return prompt

    def count_tokens(self, messages):
        try:
            encoding = tiktoken.encoding_for_model("gpt-4")
            token_count = 0
            for msg in messages:
                token_count += len(encoding.encode(json.dumps(msg['content']))) + 4
            return token_count
        except Exception:
            return sum(len(str(m['content'])) for m in messages)

    def summarize_history(self):
        # Use the model to summarize older parts of the history if it gets too long
        # We keep the last 2 messages and summarize the rest
        if len(self.messages) < 5:
            return self.messages
        
        # Prepare a summary prompt
        system_msg = {"role": "system", "content": "Summarize the following conversation history concisely, focusing on key facts and context. Keep it under 1000 tokens."}
        older_msgs = self.messages[:-2] # Keep last 2
        
        # Format older messages for summarization
        summary_input = "\n".join([f"{m['role']}: {m['content']}" for m in older_msgs])
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.get('model', 'local-model'),
                messages=[system_msg, {"role": "user", "content": f"Summarize this:\n{summary_input}"}],
                temperature=0.3
            )
            summary_text = response.choices[0].message.content
            # Replace older messages with summary
            self.messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": f"[Previous conversation summarized: {summary_text}]\n"}
            ] + self.messages[-2:]
            
            # Update history file to reflect this change? 
            # For simplicity, we just update the in-memory messages. 
            # The user asked to save history, so we should probably save the summarized version too.
            self.history = [{"role": "system", "content": self.get_system_prompt()}, 
                            {"role": "user", "content": f"[Previous conversation summarized: {summary_text}]\n"}] + self.history[-2:]
            return True
        except Exception as e:
            print(f"Error summarizing: {e}")
            return False

    def execute_tool(self, tool_call):
        func_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Human in the loop
        print(f"\nTool Call Detected: {func_name}({arguments})")
        confirm = input("Execute? (y/n): ").strip().lower()
        if confirm != 'y':
            return None

        result = ""
        if func_name == "read_file":
            filepath = arguments.get('file_path')
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    result = f.read()
            else:
                result = "File not found."
        elif func_name == "write_file":
            filepath = arguments.get('file_path')
            content = arguments.get('content')
            try:
                with open(filepath, 'w') as f:
                    f.write(content)
                result = f"File written to {filepath}."
            except Exception as e:
                result = f"Error writing file: {e}"
        elif func_name == "bash_command":
            cmd = arguments.get('command')
            try:
                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                result = proc.stdout + proc.stderr
            except Exception as e:
                result = f"Error executing command: {e}"
        elif func_name == "save_memory":
            content = arguments.get('content')
            result = self.memory.save_memory(content)
        elif func_name == "get_memory":
            query = arguments.get('query')
            result = self.memory.load_memory(query)
        elif func_name == "load_all_memory":
            result = self.memory.load_all_memory()
        
        return result

    def process_tools(self, tool_calls):
        tool_results = []
        for tc in tool_calls:
            res = self.execute_tool(tc)
            if res is not None:
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": res
                })
        return tool_results

    def run(self):
        print("Chat Interface Started. Type /help for commands.")
        
        while True:
            try:
                user_input = input("User: ").strip()
            except EOFError:
                break
            
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                parts = user_input.split()
                cmd = parts[0].lower()
                
                if cmd == "/exit":
                    print("Exiting...")
                    break
                elif cmd == "/clear":
                    self.messages = []
                    self.history = []
                    self.save_history()
                    print("History cleared.")
                elif cmd == "/history":
                    print(json.dumps(self.history, indent=2))
                elif cmd == "/memory":
                    print(self.memory.load_all_memory())
                elif cmd == "/messages":
                    print(json.dumps(self.messages, indent=2))
                elif cmd == "/shrink":
                    self.summarize_history()
                    self.save_history()
                    print("History summarized and saved.")
                elif cmd == "/threshold":
                    if len(parts) > 1:
                        try:
                            val = int(parts[1])
                            self.config.set('token_threshold', val)
                            print(f"Threshold set to {val}.")
                        except ValueError:
                            print("Invalid number.")
                    else:
                        print("Usage: /threshold <n>")
                elif cmd == "/tools":
                    print("Available tools: read_file, write_file, bash_command, save_memory, get_memory, load_all_memory")
                elif cmd == "/tokens":
                    count = self.count_tokens(self.messages)
                    print(f"Current token count: {count}")
                elif cmd == "/config":
                    print(json.dumps(self.config.data, indent=2))
                elif cmd == "/url":
                    if len(parts) > 1:
                        self.config.set('url', parts[1])
                        self.client = OpenAI(base_url=self.config.get('url'), api_key="lm-studio")
                        print(f"URL set to {parts[1]}")
                elif cmd == "/help":
                    print("""/exit - Beendet die Session.
/clear - Löscht die aktuelle Chat-Historie.
/history - Zeigt die aktuelle Historie an.
/memory - Zeigt alle gespeicherten Fakten an.
/messages - zeigt Inhalt von messages variable
/shrink - Komprimiert die Historie manuell.
/threshold <n> - Setzt das Token-Limit auf <n>.
/tools - Zeigt alle verfügbaren Tools an.
/tokens - Zeigt die aktuelle Tokenanzahl an.
/config - Zeigt die aktuelle config an.
/url - Setzt die Connection-URL.
/help - Zeigt diese Hilfe an.""")
                else:
                    print(f"Unknown command: {cmd}")
                continue

            # Add user message to history
            user_msg = {"role": "user", "content": user_input}
            self.messages.append(user_msg)
            self.history.append(user_msg)
            self.save_history()

            # Check token limit and summarize if needed
            current_tokens = self.count_tokens(self.messages)
            threshold = self.config.get('token_threshold', 5000)
            if current_tokens > threshold:
                print("Token limit reached. Summarizing history...")
                self.summarize_history()

            # Build prompt with system message
            sys_prompt = self.get_system_prompt()
            api_messages = [{"role": "system", "content": sys_prompt}] + self.messages

            try:
                response = self.client.chat.completions.create(
                    model=self.config.get('model', 'local-model'),
                    messages=api_messages,
                    temperature=self.config.get('temperature', 0.7),
                    max_tokens=self.config.get('max_tokens', 8192),
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Reads the content of a file.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {"type": "string", "description": "Path to the file"}
                                    },
                                    "required": ["file_path"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "description": "Writes content to a file.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "file_path": {"type": "string", "description": "Path to the file"},
                                        "content": {"type": "string", "description": "Content to write"}
                                    },
                                    "required": ["file_path", "content"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "bash_command",
                                "description": "Executes a bash command.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "command": {"type": "string", "description": "The bash command to execute"}
                                    },
                                    "required": ["command"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "save_memory",
                                "description": "Saves a fact to long-term memory.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "content": {"type": "string", "description": "The fact to save"}
                                    },
                                    "required": ["content"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "get_memory",
                                "description": "Retrieves memories based on a query.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {"type": "string", "description": "Search query"}
                                    },
                                    "required": ["query"]
                                }
                            }
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "load_all_memory",
                                "description": "Loads all saved memories.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {}
                                }
                            }
                        }
                    ],
                    tool_choice="auto"
                )

                assistant_msg = response.choices[0].message
                
                # Handle tool calls
                if assistant_msg.tool_calls:
                    tool_results = self.process_tools(assistant_msg.tool_calls)
                    # Add tool results to messages
                    for tr in tool_results:
                        self.messages.append(tr)
                    
                    # Call API again with tool results
                    response2 = self.client.chat.completions.create(
                        model=self.config.get('model', 'local-model'),
                        messages=api_messages + [{"role": "assistant", "tool_calls": assistant_msg.tool_calls}] + tool_results,
                        temperature=self.config.get('temperature', 0.7),
                        max_tokens=self.config.get('max_tokens', 8192)
                    )
                    final_assistant_msg = response2.choices[0].message
                else:
                    final_assistant_msg = assistant_msg

                # Format response with Rich
                if final_assistant_msg.content:
                    console = Console()
                    console.print(Panel(Markdown(final_assistant_msg.content), title="Assistant"))
                
                # Add assistant message to history
                asst_history_msg = {"role": "assistant", "content": final_assistant_msg.content}
                self.messages.append(asst_history_msg)
                self.history.append(asst_history_msg)
                self.save_history()

            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    app = ChatInterface()
    app.run()
