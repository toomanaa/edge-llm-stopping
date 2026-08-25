#!/bin/bash
# One-time setup on the Raspberry Pi 5. Run from the repo root after cloning.
set -e
sudo apt update && sudo apt install -y git build-essential cmake python3-pip unzip
if [ ! -d llama.cpp ]; then
  git config --global http.version HTTP/1.1
  git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp && cmake -B build && cmake --build build --config Release -j4 && cd ..
mkdir -p models
echo "Download a model with, e.g.:"
echo "  wget -4 -c -O models/llama-3.2-1b-instruct-q4_k_m.gguf \\"
echo "    'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf'"
echo "Then start the server:"
echo "  ./llama.cpp/build/bin/llama-server -m models/llama-3.2-1b-instruct-q4_k_m.gguf -c 2048 --port 8080"
echo "PMIC power sensor check:"
vcgencmd pmic_read_adc | head -5
