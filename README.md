# Local Research QA System on M1 Mac using LangChain and LLaMA2

## Description
This project aims to build a local QA system for scientific research based on Retrieval-Augmented Generalization (RAG). Using LangChain framework and a locally deployed LLaMA model, the system retrieves relevant content from selected research papers (PDF documents) to generate accurate and context-aware responses.

## Hardware Requirement:
🍎 Apple Silicon

Memory >= 16GB RAM

## Quick Start:
To get started with the QA system, you will need to:
- Create and using a conda environment using Miniforge3 installer
  - Install llama.cpp
  - Install Langchain
  - Install Sentence-Transformers
- Download LLM models from Huggingface

## Environment Setup:

### Downloading Miniforge3 installer:

Minimum requirements for M1 Mac: [macOS >= 11.0](https://github.com/conda-forge/miniforge?tab=readme-ov-file#requirements-and-installers)
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh 
bash Miniforge3-MacOSX-arm64.sh
```
### Activating the base environment
```
source ~/miniforge3/bin/activate
```
### Creating a sub-environment
```
conda create -n llama-env python=3.9.16
conda activate llama-env
```
### Installing Llama-cpp-python
```
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python
```
### Installing Langchain
```
pip install --upgrade --quiet langchain langchain-community langchainhub
```
To check the package installed, try:
```
pip list | grep langchain
```
### Installing Sentence-Transformers
```
pip install -U sentence-transformers
```
## Models
Llama2-7b and Llama2-13b (4 bit quantization).
### Obtaining models from Huggingface
```
huggingface-cli download TheBloke/Llama-2-13B-chat-GGUF llama-2-13b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

## To do: 
1. Try more PDF chunking methods (custom parser instead of manual markdown conversion)
2. Multi-turn chat history to support context-aware conversations.
