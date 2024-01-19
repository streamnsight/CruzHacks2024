import os
import gradio as gr

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# for Mac M series 
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

url = "https://www.techtarget.com/searchenterpriseai/tip/9-top-AI-and-machine-learning-trends"

template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Do not attempt to define what acronyms stands for unless the definition was explicitly provided in the context.
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"
"""

# load embedding model
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'mps'}
encode_kwargs = {'normalize_embeddings': False}
embedding=HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

## Note:
# temperature determines how reproducible the results will be: 0 should always get you the same answer for the same prompt, 
# 1 will be much more random
# top_n (number) or top_p (percent) determines how many of the top probable words are used in the pool of possible words. 
# smaller top_X means less randomness and potential repetitive output.

## To use local Llama model
## localtion of the downloaded model:
# from langchain_community.llms import LlamaCpp
# model_file = '/Users/emmanuel/workspace/models/llms/llama-2-13b-chat.Q4_0.gguf'
# # downloaded from https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF
# # you may also try a smaller model:
# # https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
# llm = LlamaCpp(
#     model_path=model_file,
#     n_ctx=4096, # context window
#     #verbose=True,
#     device='mps', # this is specifically to use Mac M1/2 metal GPU 
#     # model_kwargs={'device':'mps'},
#     n_gpu_layers=1,
#     temperature=0, 
#     top_p=0.95
# )


## to use OpenAI
from langchain_openai import OpenAI
llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"), temperature=0, top_p=0.95)


rag_chain = None

def ingest(url):
    global rag_chain
    # get the data
    loader = WebBaseLoader(url)
    data = loader.load()

    # split the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, 
        chunk_overlap=0, 
        length_function=len, 
    )
    all_splits = text_splitter.split_documents(data)

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding,
        # persist_directory="./"  # if you want to persist the DB locally, and not have to reindex each time
    )
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectorstore.as_retriever(search_kwargs={'k':3}),
                                       return_source_documents=True)
    rag_prompt = PromptTemplate.from_template(template)
    rag_chain = (
        {'context': qa_chain, 'question': RunnablePassthrough()}
        | rag_prompt
        | llm
    )


def generate_text(prompt):
    
    return rag_chain.invoke(prompt)

ingest(url)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                [],
                label="RAG Chat",
                show_label=True,
                show_share_button=True,
                height=500,
            )

        # with gr.Column(scale=1, offset=0.01):
        #     max_tokens = gr.Slider(
        #         256,
        #         4096,
        #         value=4096,
        #         step=16,
        #         label="max_tokens",
        #         info="The maximum number of tokens",
        #     )

        #     temperature = gr.Slider(
        #         0.0,
        #         2.0,
        #         value=0.0,
        #         step=0.1,
        #         label="temperature",
        #         info="Controls randomness in the model. The value is used to modulate the next token probabilities.",
        #     )

        #     top_p = gr.Slider(
        #         0.1,
        #         1.0,
        #         value=0.95,
        #         step=0.1,
        #         label="top_p",
        #         info="Nucleus sampling, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.",
        #     )

    with gr.Row():
        with gr.Column(scale=3):
            msg = gr.Textbox(
                label="Prompt:",
                placeholder="Type your question here",
                lines=2,
                autofocus=True,
            )

        with gr.Column(scale=1):
            send = gr.Button(value="Generate", variant="primary", scale=1)
            clear = gr.Button("Clear")

    def user(user_message, history):
        return ("", history + [[user_message, None]])

    def bot(history):
        llm_response = generate_text(
            history[-1][0]
        )

        history[-1][1] = llm_response

        return history

    send.click(
        user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(bot, [chatbot], chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

    demo.queue()
    #demo.launch(share=True)    
    demo.launch()