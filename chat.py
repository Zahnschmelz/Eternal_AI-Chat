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
            'max_tokens': 4096,
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
import json
import os
import subprocess
from datetime import datetime
from typing import List, Dict, Any

import openai
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

# --- Konfiguration & Konstanten ---
CONFIG_FILE = "config.json"
HISTORY_FILE = "chat_history.json"
MEMORY_FILE = "longterm_memory.json"

console = Console()

BLUE = "\033[94m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BROW = "\033[38;5;94m"
PURP = "\033[38;5;135m"
ORANGE = "\033[38;5;208m"
LIGHT_GREY = "\033[37m"
DARK_GRAY = "\033[90m"
BOLD_RED = "\033[38;5;88m"

class ChatAssistant:
    def __init__(self):
        self.config = self.load_config()
        self.history = self.load_history()
        self.memory = self.load_memory()
        self.client = openai.OpenAI(base_url=self.config["url"], api_key="lm-studio")
        self.token_threshold = self.config["token_threshold"]
        self.summary = ""

    def load_config(self):
        default_config = {"url": "http://localhost:1234/v1", "token_threshold": 5000, "model": "local-model"}
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return {**default_config, **json.load(f)}
        return default_config

    def save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(HISTORY_FILE, "w") as f:
            json.dump(self.history, f, indent=4)

    def load_memory(self):
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        return []

    def save_memory(self, fact: str):
        self.memory.append(fact)
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.memory, f, indent=4)

    # --- Tools ---
    def tool_read_file(self, path: str):
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    def tool_write_file(self, path: str, content: str):
        try:
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    def tool_bash_command(self, command: str):
        try:
            result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
            return result
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output}"

    def tool_save_memory(self, fact: str):
        self.save_memory(fact)
        return "Fact saved successfully."

    def tool_get_memory(self):
        return f"Current memories: {', '.join(self.memory)}" if self.memory else "Memory is empty."

    def tool_load_all_memory(self):
        return f"All memories: {json.dumps(self.memory)}"

    def ask_user_for_tool(self, func_name: str, args: dict) -> str:
        arg_str = ", ".join([f"{k}={v}" for k, v in args.items()])
        console.print(f"\n[bold cyan]Tool Call:[/bold cyan] `{func_name}({arg_str})`")
        choice = Prompt.ask("Execute this tool?", choices=["y", "n", "m"], default="y")

        if choice == "y":
            if func_name == "read_file": return self.tool_read_file(args["path"])
            if func_name == "write_file": return self.tool_write_file(args["path"], args["content"])
            if func_name == "bash_command": return self.tool_bash_command(args["command"])
            if func_name == "save_memory": return self.tool_save_memory(args["fact"])
            if func_name == "get_memory": return self.tool_get_memory()
            if func_name == "load_all_memory": return self.tool_load_all_memory()
            return ""
        elif choice == "m":
            return Prompt.ask(f"Enter manual return value for `{func_name}`")
        else:
            return f"[Skipped {func_name}]"

    # --- Logik ---
    def get_token_count(self):
        return sum(len(m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])) for m in self.history if m['role'] != 'system') // 4

    def shrink_history(self):
        if len(self.history) <= 2: return
        dialogue = [m for m in self.history if m['role'] in ['user', 'assistant']]
        if len(dialogue) <= 2: return
        to_summarize = dialogue[:-2]
        remaining = dialogue[-2:]
        prompt = f"Summarize the following conversation history into a single concise paragraph: {json.dumps(to_summarize, default=str)}"
        response = self.client.chat.completions.create(model=self.config["model"], messages=[{"role": "user", "content": prompt}])
        self.summary = response.choices[0].message.content
        self.history = [{"role": "system", "content": f"Summary of previous chat: {self.summary}"}] + remaining

    def run_chat(self):
        session = PromptSession()
        custom_style = Style.from_dict({'prompt_color': 'cyan'})
        commands = ["/exit", "/clear", "/history", "/memory", "/shrink", "/threshold", "/tools", "/tokens", "/config", "/url", "/help"]
        completer = WordCompleter(commands)
        console.print(Panel("[bold green]Eternal AI-Chat[/bold green]\nType /help for commands."))


        while True:
            try:
                #user_input = session.prompt("\033[36m> \033[0m", completer=completer).strip()
                print()
                user_input = session.prompt(
                    HTML('<prompt_color>> </prompt_color>'),
                    completer=completer,
                    style=custom_style
                ).strip()
                if not user_input: continue
                if user_input == "/exit":
                    self.save_history()
                    break
                elif user_input == "/clear":
                    self.history = []
                    self.summary = ""
                    console.print("[yellow]History cleared.[/yellow]")
                elif user_input == "/history":
                    for m in self.history:
                        role_color = "bold cyan" if m['role'] == 'user' else "bold magenta"
                        content = m['content'] if isinstance(m['content'], str) else json.dumps(m['content'])
                        console.print(f"[{role_color}]{m['role'].upper()}[/ {role_color}]: {content}")
                elif user_input == "/memory":
                    console.print(f"[blue]Memories:[/blue]\n{self.memory}")
                elif user_input == "/shrink":
                    self.shrink_history()
                    console.print("[green]History compressed.[/green]")
                elif user_input.startswith("/threshold"):
                    self.token_threshold = int(user_input.split()[1])
                    self.config["token_threshold"] = self.token_threshold
                    self.save_config()
                elif user_input == "/tokens":
                    console.print(f"Current estimated tokens: {self.get_token_count()}")
                elif user_input == "/config":
                    console.print(self.config)
                elif user_input.startswith("/url"):
                    self.config["url"] = user_input.split()[1]
                    self.client = openai.OpenAI(base_url=self.config["url"], api_key="lm-studio")
                    self.save_config()
                elif user_input == "/tools":
                    console.print("Available tools: `read_file`, `write_file`, `bash_command`, `save_memory`, `get_memory`, `load_all_memory`")
                elif user_input == "/help":
                    console.print("Commands: /exit, /clear, /history, /memory, /shrink, /threshold <n>, /tools, /tokens, /config, /url <url>, /help")
                else:
                    print()
                    self.history.append({"role": "user", "content": user_input})
                    if self.get_token_count() > self.token_threshold:
                        self.shrink_history()
                    self.process_message()
                    self.save_history()

            except KeyboardInterrupt: break
            except Exception as e: console.print(f"[red]Error: {e}[/red]")

    def process_message(self):
        now = datetime.now().strftime("%H:%M:%S")
        sys_msg = (
            "You are a professional system assistant. "
            "You can read/write files, run bash commands, and manage a long-term memory. "
            "Automatically store relevant facts in your long-term memory. "
            f"Current time: {now}. "
            "Use tools if needed. Work efficiently."
        )

        tools = [
            {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "bash_command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "save_memory", "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}}},
            {"type": "function", "function": {"name": "get_memory", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "load_all_memory", "parameters": {"type": "object", "properties": {}}}}
        ]

        current_messages = [{"role": "system", "content": sys_msg}] + self.history

        with console.status("[bold blue]Thinking...", spinner="dots") as status:
            while True:
                response = self.client.chat.completions.create(
                    model=self.config["model"],
                    messages=current_messages,
                    tools=tools,
                    tool_choice="auto"
                )

                msg = response.choices[0].message

                if not msg.tool_calls:
                    if msg.content:
                        self.history.append({"role": "assistant", "content": msg.content})
                        console.print(Markdown(msg.content))
                    break

                assistant_msg_dict = {
                    "role": "assistant",
                    "content": msg.content if msg.content else "",
                    "tool_calls": [
                        {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in msg.tool_calls
                    ]
                }

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments)
                    result = self.ask_user_for_tool(func_name, args)

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result)
                    }

                    current_messages.append(tool_msg)

if __name__ == "__main__":
    assistant = ChatAssistant()
    assistant.run_chat()
