import os
import json
import subprocess
import datetime
import tiktoken
import base64
import mimetypes
from typing import List, Any, Dict, Union

from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm

import pygame

# --- Konfiguration & Dateipfade ---
CONFIG_FILE = "config.json"
HISTORY_FILE = "history.json"
MEMORY_FILE = "memory.json"
SOUND_FILE = "answer.mp3"

console = Console()

class Config:
    def __init__(self):
        self.defaults = {
            "url": "http://localhost:1234/v1",
            "api_key": "lm-studio",
            "temperature": 0.7,
            "token_threshold": 5000,
            "model": "model-identifier"
        }
        self.data = self.load()

    def load(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    loaded = json.load(f)
                    return {**self.run_defaults(), **loaded} # Fix logic
            except Exception:
                return self.defaults
        return self.defaults

    def run_defaults(self):
        return self.defaults

    def save(self):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(self.data, f, indent=4)

    def update(self, key, value):
        self.data[key] = value
        self.save()

class MemoryManager:
    def __init__(self, path):
        self.path = path
        self.memory = self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def add(self, fact: str):
        self.memory.append({"fact": fact})
        self.save()

    def get_all_formatted(self):
        if not self.memory:
            return "Memory is empty."
        lines = [f"{m.get('fact', '')}" for m
                 in self.memory]
        return "\n".join(lines)

class HistoryManager:
    def __init__(self, path, config: Config):
        self.path = path
        self.config = config
        self.messages = self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.messages, f, indent=4)

    def add_message(self, role: str, content: Any, tool_calls: List[Dict] = None,
                    client: OpenAI = None, system_prompt: str = None, memories_text: str = ""):
        #msg = {"role": role, "content": str(content) if content else ""}
        msg = {"role": role, "content": content if content else ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)
        self.save()

        if client and system_prompt:
            self.check_buffer_summary(client, system_prompt, memories_text)

    def check_buffer_summary(self, client: OpenAI, system_prompt: str, memories_text: str):
        total = self.get_total_tokens()
        threshold = self.config.data['token_threshold']

        if total > threshold and len(self.messages) > 4:
            console.print("[yellow]Token limit reached. Summarizing history via LLM...[/yellow]")
            self.shrink(client, system_prompt, memories_text)

    def shrink(self, client: OpenAI, system_prompt: str, memories_text: str):
        if len(self.messages) <= 1:
            return

        history_text = ""
        for m in self.messages:
            role = m['role']
            content = m.get('content', '')
            # Falls content eine Liste ist (Vision), extrahieren wir nur den Text
            if isinstance(content, list):
                text_part = "".join([item.get('text', '') if isinstance(item, dict) and item.get('type') == 'text' else "" for item in content])
                history_text += f"{role}: {text_part}\n"
            else:
                history_text += f"{role}: {str(content)}\n"

        summary_prompt = (
            "Summarize the following conversation history into a concise summary. "
            "The summary should be informative but brief, aiming for around 1000 tokens maximum. "
            "Focus on key facts, decisions, and context.\n\n"
            f"History:\n{history_text}"
        )

        try:
            response = client.chat.completions.create(
                model=self.config.data['model'],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user", "content": summary_prompt}
                ]
            )
            summary_text = response.choices[0].message.content

            keep_messages = self.messages[-2:]
            self.messages = [
                {"role": "system", "content": f"Summary of previous conversation: {summary_text}"},
                {"role": "system", "content": f"Long-term Memories:\n{memories_text}"}
            ] + keep_messages

            self.save()
            console.print("[green]History summarized and memories re-attached.[/green]")
        except Exception as e:
            console.print(f"[red]Failed to summarize history: {str(perm_error(e))}[/red]")

    def clear(self):
        self.messages = []
        if os.path.exists(self.path):
            os.remove(self.path)

    def count_tokens(self, text: str):
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except:
            return len(text) // 4

    def get_total_tokens(self):
        total = 0
        for m in self.messages:
            if isinstance(m, dict) and 'content' in m:
                total += self.count_tokens(m['content'])
        return total

# Helper functions for the error handling logic used above
def string_content(c): return str(c)
def perm_error(e): return str(e)

class CommandCompleter(Completer):
    def get_completions(self, document, complete, *args):
        text = document.text
        if not text.startswith('/'):
            return
        commands = ['/exit', '/clear', '/history', '/memory', '/shrink', '/threshold', '/tools', '/tokens', '/config', '/url', '/help']
        for cmd in commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text))

class ChatInterface:
    def __init__(self):
        try:
            print("HUI")
            pygame.mixer.init()
            self.sound_available = os.path.exists(SOUND_FILE)
            if not self.sound_available:
                console.print(f"[yellow]Warning: {SOUND_FILE} not found.[/yellow]")
        except Exception as e:
            console.print(f"[red]Sound error: {e}[/red]")
            self.sound_available = False

        os.system('clear')

        self.config = Config()
        self.memory = MemoryManager(MEMORY_FILE)
        self.history = HistoryManager(HISTORY_FILE, self.config)
        self.client = OpenAI(base_url=self.config.data['url'], api_key=self.config.data['api_key'])
        self.session = PromptSession(completer=CommandCompleter())

    def play_answer_sound(self):
        if self.sound_available:
            try:
                pygame.mixer.music.load(SOUND_FILE)
                pygame.mixer.music.play()
            except Exception:
                pass

    def get_system_prompt(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            "You are a professional system assistant. "
            "You can read/write files, run bash commands, and manage a long-term memory. "
            "If the user tells you a fact about themselves or the system, use 'save_memory' to remember it. "
            "If you need to recall a known fact, use 'load_all_memory'. "
            "Don't just make things up. "
            "Use the available tools if they help you solve a problem. "
            "Work efficiently by chaining commands together. "
            "Always provide clear and professional answers. "
            f"\n\nCurrent Date/Time: {now}"
        )

    # --- Tools ---
    def tool_read_file(self, path: str):
        with open(path, 'r', encoding='utf-8') as f: return f.read()

    def tool_write_file(self, path: str, content: str):
        with open(path, 'w', encoding='utf-8') as f: f.write(content)
        return f"File {path} written."

    def tool_bash_command(self, command: str):
        try:
            return subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        except subprocess.CalledProcessError as e:
            return f"Error: {e.output}"

    def tool_save_memory(self, fact: str):
        self.memory.add(fact)
        return "Fact saved."

    def tool_load_all_memory(self):
        return self.memory.get_all_formatted()

    def tool_describe_image(self, path: str):
        """Liest das Bild ein und gibt den Base64-String zurück, damit das Modell es 'sieht'."""
        try:
            if not os.path.exists(path):
                return f"Error: File {path} not found."

            mime_type, _ = mimetypes.guess_type(path)
            if not mime_type or not mime_type.startswith("image/"):
                return "Error: File is not a valid image."

            with open(path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return f"data:{mime_type};base64,{encoded_string}"
        except Exception as e:
            return f"Error reading image: {str(e)}"

    def execute_tool(self, tool_name, args):
        console.print(Panel(f"[bold cyan]Proposed Tool Call:[/bold cyan]\n{tool_name}({args})"))
        if not Confirm.ask("Allow execution?"):
            return "Denied."
        try:
            if tool_name == "read_file": return self.tool_read_file(args['path'])
            if tool_name == "write_file": return self.tool_write_file(args['path'], args['content'])
            if tool_name == "bash_command": return self.tool_bash_command(args['command'])
            if tool_name == "save_memory": return self.tool_save_memory(args['fact'])
            if tool_name == "get_memory": return self.tool_get_memory(args['query'])
            if tool_name == "load_all_memory": return self.tool_load_all_memory()
            if tool_name == "describe_image": return self.tool_describe_image(args['path'])
            return "Unknown tool."
        except Exception as e:
            return f"Error: {str(e)}"

    def run(self):
        console.print(Panel("[bold green]Eternal_AI-Chat Started[/bold green]\nType /help for commands."))

        tools_definition = [
            {"type": "function", "function": {"name": "read_file", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "bash_command", "description": "Run bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "save_memory", "description": "Save fact", "parameters": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]}}},
            {"type": "function", "function": {"name": "load_all_memory", "description": "Load all", "parameters": {"type": "object", "properties": {}}}},
            {"type": "function", "function": {"name": "describe_image", "description": "Read an image file and return its Base64 data so the model can see it.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
        ]

        while True:
            try:
                user_input = self.session.prompt("\n>> ")
                if not user_input: continue

                sys_prompt = self.get_system_prompt()
                memories_str = self.memory.get_all_formatted()

                if user_input.startswith('/'):
                    parts = user_input.split(' ', 1)
                    cmd = parts[0]
                    arg = parts[1] if len(parts) > 1 else None
                    if cmd == '/exit': break
                    elif cmd == '/clear': self.history.clear(); console.print("Cleared.")
                    elif cmd == '/memory': console.print(Panel(self.memory.get_all_formatted(), title="Memory"))
                    elif cmd == '/tokens': console.print(f"Tokens: {self.history.get_total_tokens()}")
                    elif cmd == '/config': console.print(self.config.data)
                    elif cmd == '/shrink': self.history.shrink(self.client, sys_prompt, memories_str)
                    elif cmd == '/url' and arg:
                        self.config.update('url', arg)
                        self.client = OpenAI(base_url=arg, api_key=self.config.data['api_key'])
                    elif cmd == '/history':
                        if not self.history.messages:
                            console.print("History is empty.")
                        else:
                            for msg in self.history.messages:
                                role = msg['role']
                                content = msg.get('content', '')
                                console.print(f"[{role}]: {content}")
                    elif cmd == '/threshold':
                        console.print(f"Token Threshold: {self.config.data['token_threshold']}")
                    elif cmd == '/tools':
                        console.print(Panel(json.dumps(self.tools_definition, indent=2), title="Available Tools"))
                    elif cmd == '/help':
                        help_text = (
                            "/exit - Exit the chat\n"
                            "/clear - Clear chat history\n"
                            "/history - Show chat history\n"
                            "/memory - Show long-term memory\n"
                            "/shrink - Summarize history\n"
                            "/threshold - Show token threshold\n"
                            "/tools - Show available tools\n"
                            "/tokens - Show current token count\n"
                            "/config - Show configuration\n"
                            "/url <url> - Update API URL\n"
                            "/help - Show this help message"
                        )
                        console.print(Panel(help_text, title="Help Menu"))
                    continue

                print()

                self.history.add_message("user", user_input, client=self.client, system_prompt=sys_prompt, memories_text=memories_str)

                while True:
                    api_messages = [{"role": "system", "content": sys_prompt}] + self.history.messages

                    response = self.client.chat.completions.create(
                        model=self.config.data['model'],
                        messages=api_messages,
                        tools=tools_definition,
                        temperature=self.config.data['temperature']
                    )

                    resp_msg = response.choices[0].message

                    if resp_msg.tool_calls:
                        tc_list = []
                        for tc in resp_msg.tool_calls:
                            tc_list.append({
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                            })
                        self.history.add_message("assistant", resp_msg.content or "", tool_calls=tc_list, client=self.client, system_prompt=sys_prompt, memories_text=memories_str)

                        for tc in resp_msg.tool_calls:
                            func_name = tc.function.name
                            args = json.loads(tc.function.arguments)
                            result = self.execute_tool(func_name, args)
                            if func_name == "describe_image" and result.startswith("data:image/"):
                                self.history.add_message("tool", f"Image loaded successfully from {args['path']}", client=self.client, system_prompt=sys_prompt, memories_text=memories_str)
                                # Wir fügen eine neue User-Nachricht mit dem eigentlichen Bildinhalt hinzu
                                self.history.messages.append({
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": f"Here is the image from the tool call: {args['path']}"},
                                        {"type": "image_url", "image_url": {"url": result}}
                                    ]
                                })
                            else:
                                self.history.add_message("tool", result, client=self.client, system_prompt=sys_prompt, memories_text=memories_str)
                        continue
                    else:
                        ans = resp_msg.content or ""
                        self.history.add_message("assistant", ans, client=self.client, system_prompt=sys_prompt, memories_text=memories_str)
                        console.print(Markdown(ans))
                        self.play_answer_control(ans)
                        break

            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")

    def play_answer_control(self, ans):
        self.play_answer_sound()

if __name__ == "__main__":
    chat = ChatInterface()
    chat.run()
