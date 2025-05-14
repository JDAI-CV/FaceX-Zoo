#!/bin/bash

echo "Sprawdzanie środowiska..."

OS="$(uname -s)"

if [ "$OS" == "Darwin" ]; then
    ACTIVATE=".venv/bin/activate"
    PYTHON=python3
elif [ "$OS" == "Linux" ]; then
    ACTIVATE=".venv/bin/activate"
    PYTHON=python3
elif [[ "$OS" == MINGW* || "$OS" == CYGWIN* || "$OS" == MSYS* ]]; then
    ACTIVATE=".venv/Scripts/activate"
    PYTHON=python
else
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Tworzenie środowiska virtualenv..."
    $PYTHON -m venv .venv
fi

echo "Aktywacja środowiska..."
source "$ACTIVATE"

echo "Instalacja zależności..."
pip install --upgrade pip
pip install --break-system-packages -r requirements.txt

echo "Uruchamianie systemu..."
$PYTHON src/main.py
