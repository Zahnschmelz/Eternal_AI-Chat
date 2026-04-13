#!/bin/bash

# Pfad zum Verzeichnis des Skripts (wichtig für venv-Erstellung im richtigen Ordner)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 1. Check ob venv existiert, sonst erstellen
if [ ! -d "venv" ]; then
    echo "Erzeuge Virtual Environment (venv)..."
    python3 -m venv venv
fi

# 2. venv aktivieren
echo "Aktiviere venv..."
source venv/bin/activate

# 3. Requirements installieren
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Konsole leeren
echo "Running the application..."
clear

# 5. Chat starten
python3 chat.py
