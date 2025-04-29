from dotenv import find_dotenv, load_dotenv
import box
import yaml

from llama_cpp import Llama
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
# from transformers import AutoModelForCausalLM

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])



# # Load the LLaMA 2 7B model
model_name = "./models/llama-2-7b-chat.Q4_K_M.gguf"  # model name
# llm_model = Llama(model_name)
n_gpu_layers = 1 # good enough for Metal 
n_batch = 128#128 # should be between 1 and n_ctx, considering RAM on Apple Silicon
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# model_name = "./models/functionary-small-v2.4.Q4_0.gguf"
def langchain_llm(): # load model using langchain
    llm = LlamaCpp(
        model_path = model_name, 
        n_gpu_layers=n_gpu_layers, # number of layers to be loaded into gpu memory
        n_batch=n_batch, # Number of tokens to process in parallel
        n_ctx = 2048, #token context window
        f16_kv=True, # Must be True
        callbacks = callback_manager,
        verbose=True,
        max_tokens=1024, # the maximum number of tokens to generate
        # chat_format="chatml-function-calling"
    )
    return llm


def llamacpp_llm(): # load model using llama-cpp-python
    llm = Llama(
        model_path = model_name, 
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx = 2048,
        f16_kv=True,
        verbose=True,
        chat_format="chatml"
    )
    return llm

if __name__ == "__main__":
    
    # print(llm_model)
    questions = "Can you name the planets in solar system?"

    # llm = llamacpp_llm()
    # output = llm(
    #     "Q: {questions} A: ", # Prompt
    #     # max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
    #     stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
    #     echo=True # Echo the prompt back in the output
    # ) # Generate a completion, can also call create_completion
    # print(output)

    
    llm = langchain_llm()
    # # questions = "Q: How long do I get to the moon if i walk from the earth? Let's think step by step. A:"
    # # questions = "what is llama?"
    # # questions = "what is jordan normal form?"
    # # questions = "What attention mechanisms does Llama 2 7b use?"
    # # questions = "what Transformer does Llama 2 use?"
    
    # for ctx_len in [512, 1024, 2048]:
    #     # messages = llm.invoke(questions)
    #     try: llm.create_chat_completion(output, max_tokens=ctx_len)
    #     except Exception as e:
    #         print(f"failed at token length {ctx_len}: {str(e)}")
    
    llm.invoke("how many species of plants in North America? Please show me the most common ones, including the name and the amount.")
