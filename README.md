# Local Research QA System on M1 Mac using LangChain and LLaMA2

## Description
A local QA system for scientific research, powered by LLMs running locally on an M1 Mac. 

## Context: 
Large Language Models (LLMs) are powerful at understanding human instructions. However, they often struggle with domain-specific queries, especially in areas like Advanced Mathematics, Scientific Research, and Engineering. 

To better handle these topics, LLMs require access to domain knowledge as external context. 

This project aims to build a local QA system for scientific research based on Retrieval-Augmented Generalization (RAG) framework. Using LangChain and a locally deployed LLaMA model, the system retrieves relevant content from selected research documents (such as PDFs) to generate accurate and context-aware responses.

## Hardware Requirement:
🍎 M1 Mac with 16GB RAM

## Environment Setup:
1. Installing Miniforge3 for MacOS:
```
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh 
bash Miniforge3-MacOSX-arm64.sh
```
2. Pointing to the environment path
```
source ~/miniforge3/bin/activate
```
3. Creating a Python environment
```
conda create -n llama-env python=3.9.16
conda activate llama-env
```
4. Installing Llama-cpp-python
```
pip uninstall llama-cpp-python -y
conda create -n llama python=3.9.16 [](https://llama-cpp-python.readthedocs.io/en/latest/install/macos/#__codelineno-2-2)conda activate llama
```
5. Installing Langchain
```
pip install --upgrade --quiet langchain langchain-community langchainhub
```
To check the package installed, try:
```
pip list | grep langchain
```

## To do: 
1. Implement improved chunk splitting for PDF documents (custom parser instead of manual markdown conversion)
2. Multi-turn chat history to support context-aware conversations.
