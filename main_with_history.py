import box
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from typing import List
from typing_extensions import Annotated, TypedDict
from operator import itemgetter
import pprint

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# from online_retrievers import retriever
from src.model import langchain_llm
from src.prompts import qa_system_message_content

import sys

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

llm = langchain_llm()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs ={"device": "mps"})
### Construct Retriever ###
vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

# Retriever
s_k = {'k': cfg.VECTOR_COUNT}  #similarity_score top-k ranking
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs=s_k)
print(f"!!!! RETRIEVER TYPE = {type(retriever)}")

### Provide contextualized queries ###
contextualized_q_system_prompt = """
You are an assistant for question-answering tasks.
Please summarize the following pieces of retrieved context to answer the question. 
Do not generate user's questions or repeat chat labels.
If you don't know the answer, just say that you don't know.
Only return helpful answer and nothing else.

Helpful Answer: 
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualized_q_system_prompt), 
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

### Question Answering ###
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_message_content), 
        MessagesPlaceholder("chat_history"), 
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Manage Chat History with States ###
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain, 
    get_session_history, 
    input_messages_key="input", 
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# clean history
def clean_history(history):
    """Remove role labels in chat history"""
    return [msg.replace("\nHuman:", "").replace("\nAi:", "").replace("\nAssistant") for msg in history]


if __name__ == "__main__":
    print(sys.executable)
    pprint.pp(store)
    chat_history = get_session_history("test123")
    chat_history.clear()
    chat_history.add_user_message(("Can you tell me what is Llama 2?"))
    chat_history.add_ai_message(("Llama 2 is a language model developed by the authors of this paper as part of their effort to advance the responsible development of large language models (LLMs). They share observations they made during the development of Llama 2 and Llama 2-Chat, including the emergence of tool usage and temporal organization of knowledge."))
    chat_history.add_user_message(("What is the main goal of Llama 2?"))
    chat_history.add_ai_message(("According to the authors, the main goal of Llama 2 is to advance the responsible development of large language models (LLMs). They aimed to create a new family of LLMs that can be used for various natural language processing tasks."))
    chat_history.add_user_message(("Can you tell me how it can be used?"))
    chat_history.add_ai_message(("Llama 2 can be used for a variety of applications, such as language translation, text summarization, and chatbots. Its ability to generate coherent and contextually appropriate text makes it well-suited for tasks that require the creation or processing of natural language text. Additionally, its ability to learn from large amounts of data means that it can be trained on a wide range of topics and domains, making it a versatile tool for a variety of applications."))
        

    conversational_rag_chain.invoke(
        {"input": "What is the architecture of it?"}, 
        config = {
            "configurable": {"session_id": "test123"}
        },
    )["answer"]

    pprint.pp(store)

    # from langchain_core.messages import HumanMessage
    # chat_history = []
    # # to do : return_source_documents
    # # question = "Can you explain why in Chapter 4, the Hessain matrix can determine whether the critical point has a optimal value?"
    # # question = "Can you list all types of attentions that llama 2 use?"
    # # question = "Can you list the context lengths of llama 1 and llama 2 models?"
    # question = "What is Llama 2?"
    # ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
    # chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

    # second_question = "What are the fine-tuning steps of it?"
    # ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})
    # # chat_history.extend([HumanMessage(content=second_question), ai_msg_2["answer"]])
    # # qa_chain.invoke(question)
    # print(ai_msg_2["answer"])
   