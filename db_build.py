"""
Vector Database from PDF documents
including: Retrievers and vector store

text splitting ---> retriever (embeddings) ---> vector store (FAISS)
"""
import box
import yaml
import pprint
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
# from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import UnstructuredMarkdownLoader, UnstructuredPDFLoader

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# load documents
def load_documents(is_md: bool):
    if is_md: # load MD documents
        loader = UnstructuredMarkdownLoader(file_path=cfg.DATA_PATH + "Llama2_pg1_20.md")
    else:
        loader = DirectoryLoader(path=cfg.DATA_PATH, 
                                glob='*.pdf',
                                loader_cls=PyMuPDFLoader)
    documents = loader.load()
    return documents

def run_db_build(documents, is_md: bool):
    
    # MD splits (with char-level)
    if is_md == True: 
        md_text_splitter = MarkdownTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                chunk_overlap=cfg.CHUNK_OVERLAP)
        
        texts = md_text_splitter.split_documents(documents)
    
    else:
        # Char-level text splits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=cfg.CHUNK_SIZE,
                                                   chunk_overlap=cfg.CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)

    ## retriever (embeddings)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'mps'})
    # embeddings = SentenceTransformer("all-miniLM-L6-v2")
    ## vector store (FAISS)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(cfg.DB_FAISS_PATH)



if __name__ == "__main__":
    documents = load_documents(is_md = True)
    
    # print(f'!!! Metadata = {documents.meta_data}')
    print(f'!!!doc type = {type(documents)}')
    doc_content = documents[0].page_content
    print(doc_content[:250])

    ## get the title of md document
    headers_to_split_on = [("#", "Header 1")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, 
                                                   strip_headers=False,
                                                   return_each_line=True)
    md_header_splits =markdown_splitter.split_text(doc_content)

    print(md_header_splits[0].page_content)


    # text splitting
    # run_db_build(documents, is_md=True)
