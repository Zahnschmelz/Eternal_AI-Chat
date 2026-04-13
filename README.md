# eternal_chat
general purpose chat shell (local LLM)

interactive chat_shell to interact via pythons openai module with local LLMs like Ollama or lm-studio thrugh terminal, providing tool/function calling and simple tts/stt. (using arch)

prerequisires 

"""

sudo pacman -Syu && sudo pacman -Sy konsole git wget curl

git clone https://github.com/Zahnschmelz/gp_chat_shell.git

cd gp_chat_shell

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt && pip install -e .


"""
