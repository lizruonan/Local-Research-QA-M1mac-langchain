# Local Research QA System on M1 Mac using LangChain and LLaMA2 

## Description
A local QA system for scientific research, powered by LLMs running locally on an M1 Mac. 

## Context: 
Large Language Models (LLMs) are powerful at understanding human instructions. However, they often struggle with domain-specific queries, especially in areas like Advanced Mathematics, Scientific Research, and Engineering. 

To better handle these queries, LLMs require access to domain knowledge as external context. 

This project aims to build a local QA system for scientific research based on Retrieval-Augmented Generalization (RAG) framework. Using LangChain and a locally deployed LLaMA2 model, the system retrieves relevant content from selected research documents (such as PDFs) to generate accurate and context-aware responses.

## Next Step: 
1. Implement improved chunk splitting for PDF documents (custom parser instead of manual markdown conversion)
2. Add multi-turn chat history to support context-aware conversations.
