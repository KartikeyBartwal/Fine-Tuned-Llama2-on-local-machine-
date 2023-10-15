from langchain import PromptTemplate 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers 
from langchain.chains import RetrievalQA 
import os
import chainlit as cl
os.environ["REPLICATE_API_TOKEN"] = "r8_1tXV1Jxf2JBIiunu0PrGCEuYgOabv251OjQ0x"

from langchain.llms import Replicate
from langchain import PromptTemplate, LLMChain


DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = ''' Use the following pieces of information to answer the user's question. 

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else
'''


def set_custom_prompt():
    '''
    prompt template for QA retrieval for each vector stores
    '''

    prompt = PromptTemplate(template = custom_prompt_template , input_variables = ['context' , 'question'])

    return prompt 

def load_llm():
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.75,
           "max_length": 500,
           "top_p": 1},
        )
    return llm

def retrieval_qa_chain(llm , prompt , db):
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm ,
        chain_type = "stuff" ,
        retriever = db.as_retriever(search_kwargs = {'k' : 2}),
        return_source_documents = True , 
        chain_type_kwargs = {'prompt' : prompt}
    )

    return qa_chain

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2' , model_kwargs = {'device' : 'cpu'})

    db = FAISS.load_local(DB_FAISS_PATH , embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm , qa_prompt , db)

    return qa


def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query' : query})
    return response



#################### Time for ChainLit #######################################################

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content = "Starting your AI...")
    await msg.send()
    msg.content = "Hi, I am an Oliver Twist nerd. You can ask me all the stuffs related to the novel "
    await msg.update()
    cl.user_session.set("chain" , chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer= True, answer_prefix_tokens=["FINAL" , "ANSWER"]
    )
    cb.answer_reached = True 
    res = await chain.acall(message , callbacks = [cb])
    answer = res["result"]
    sources = res["source_documents"]

    # if(sources):
    #     answer += "Yes there are sources :)"
    # else:
    #     answer += f"\n No sources found"

    await cl.Message(content = answer).send()




