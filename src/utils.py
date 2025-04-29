import box
import yaml

from langchain_core.prompts.prompt import PromptTemplate
# from langchain.chains import RetrievalQA
# from langchain.chains import create_retrieval_chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.prompts import prompt_template
from src.model import langchain_llm

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

def set_prompts():
    PROMPT = PromptTemplate(input_variables=["context", "question"],
                            template=prompt_template)
    return PROMPT
