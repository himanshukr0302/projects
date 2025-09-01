from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, GEMINI_CHAT_MODEL

def build_llm(llm_choice: str):
    if llm_choice == "OpenAI":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set")
        return ChatOpenAI(model=OPENAI_CHAT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
    return ChatGoogleGenerativeAI(model=GEMINI_CHAT_MODEL, temperature=0)

def build_qa_chain(vectordb, llm_choice: str):
    llm = build_llm(llm_choice)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever), llm
