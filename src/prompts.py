from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know.

Context : {context}
Question: {question}

Only return helpful answers and nothing else.
Helpful Answer:
"""
qa_prompt = PromptTemplate(input_variables=["context", "question"],
                            template=prompt_template)

qa_system_message_content = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Use three sentences maximum and keep the answer concise.
\n\n
f{context}

Only return helpful answers and nothing else.
Helpful Answer: 
"""


prompt_template_with_history = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI does not know the answer to a question, then truthfully says it does not know.
Current conversation: {chat_history}
Human: {input}

AI:
"""
qa_prompt_with_history = ChatPromptTemplate.from_messages(
    [
        (
            prompt_template_with_history
        ), 
        MessagesPlaceholder(variable_name="history"), 
        ("human", "{input}")
    ]
)