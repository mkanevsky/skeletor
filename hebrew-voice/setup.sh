#!/usr/bin/env bash
# Reproduce the environment (Python 3.12 venv via uv).
set -e
cd "$(dirname "$0")"
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install faster-whisper edge-tts soundfile numpy scipy pedalboard onnxruntime huggingface_hub openai httpx sherpa-onnx sounddevice
uv pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary llama-cpp-python
git clone --depth 1 https://github.com/maxmelichov/BlueTTS.git third_party_BlueTTS && uv pip install -e third_party_BlueTTS
hf download notmax123/BlueTTS2.5-onnx --repo-type model --local-dir onnx_models
hf download thewh1teagle/zipformer2-66m-streaming-he --local-dir zipformer_he
mkdir -p vad && curl -sL -o vad/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx  # live.py VAD (auto-downloaded too)
# live.py on Linux also needs PortAudio: sudo apt install libportaudio2   (macOS/Windows: bundled with the sounddevice wheel)
hf download unsloth/gemma-4-E2B-it-GGUF gemma-4-E2B-it-Q4_K_M.gguf --local-dir llm
hf download unsloth/gemma-3-4b-it-GGUF gemma-3-4b-it-Q4_K_M.gguf --local-dir llm      # optional
hf download ivrit-ai/whisper-large-v3-turbo-ct2                                        # optional (needs GPU to be fast)
