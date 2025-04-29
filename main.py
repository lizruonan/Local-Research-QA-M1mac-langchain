import box
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from typing import List
from typing_extensions import Annotated, TypedDict
from operator import itemgetter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# from online_retrievers import retriever
from src.model import langchain_llm, llamacpp_llm
from src.prompts import qa_prompt

import sys

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs ={"device": "mps"})
# retrieval_chain = create_retrieval_chain(embeddings,)
vectordb = FAISS.load_local(cfg.DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
# llm = langchain_llm()
llm = llamacpp_llm()


# def format_docs(docs: List[Document]) -> str:
#     """Convert docs into strings"""
#     formatted = [
#         f"Title: {doc[0]}\nAbstract: {doc.page_content}"
#         for doc in docs
#     ]
#     return "\n\n" + "\n\n".join(formatted)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Retriever
s_k = {'k': cfg.VECTOR_COUNT}  #similarity_score top-k ranking
retriever = vectordb.as_retriever(search_type="mmr", search_kwargs=s_k)

# sub-chain, for
format = itemgetter("docs") | RunnableLambda(format_docs)

# Complete chain
qa_chain = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough(),
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)

qa_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | qa_prompt
    | llm
    | StrOutputParser()
)
qa_chain_with_source = RunnableParallel(
    {
        "context": retriever, 
        "question": RunnablePassthrough()
     }
).assign(answer=qa_chain_from_docs)



if __name__ == "__main__":
    print(sys.executable)
    # to do : return_source_documents
    # question = "Can you explain why in Chapter 4, the Hessain matrix can determine whether the critical point has a optimal value?"
    # question = "Can you list all types of attentions that llama 2 use?"
    question = "Can you tell me how can Llama 2 be used?" #"Can you list the context lengths of llama 1 and llama 2 models?"
    # qa_chain.invoke(question)
    qa_chain_with_source.invoke(question)