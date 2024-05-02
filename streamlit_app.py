import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
sys.path.append("../C3 搭建知识库") # 将父目录放入系统路径中
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
from zhipuai_llm import ZhipuAILLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.pdf import PyMuPDFLoader
from io import BytesIO
import fitz
from langchain_core.documents.base import Document

def generate_response(input_text, openai_api_key):
    llm = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=openai_api_key)
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    #st.info(output)
    return output

def get_vectordb(pdf_contents,api_key):
    # 定义 Embeddings
    embedding = ZhipuAIEmbeddings(newapi_key=api_key)
    doc = pdf_contents
    text = ""
    for page in doc:
        text += page.get_text()

    
    
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=50)

    split_docs = text_splitter.split_documents(text)
    vectordb = Chroma.from_documents(
    documents=split_docs[:20], # 为了速度，只选择前 20 个切分的 doc 进行生成；使用千帆时因QPS限制，建议选择前 5 个doc
    embedding=embedding
)
  
    return vectordb

#带有历史记录的问答链
def get_chat_qa_chain(question:str,openai_api_key:str):
    vectordb = get_vectordb()
    llm = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=openai_api_key)
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    retriever=vectordb.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    result = qa({"question": question})
    return result['answer']

#不带历史记录的问答链
def get_qa_chain(question:str,openai_api_key:str,vectordb):
    
    llm = ZhipuAILLM(model="chatglm_std", temperature=0, api_key=openai_api_key)
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
        案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
        {context}
        问题: {question}
        """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]


# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 简单大模型应用开发')
    zhipuai_api_key = st.sidebar.text_input('ZHIPUAI API KEY', type='password')
    uploaded_file = st.file_uploader("上传PDF文件", type="pdf")

    # 如果用户上传了文件
    if uploaded_file is not None:
        pdf_data = BytesIO(uploaded_file.read())
        pdf_contents = fitz.open("pdf", pdf_data)
        vectordb = get_vectordb(pdf_contents,zhipuai_api_key)
    

    # 添加一个选择按钮来选择不同的模型
    #selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式(在这里读取你的pdf文档)", "带历史记录的检索问答模式"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt, zhipuai_api_key)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt,zhipuai_api_key,vectordb)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt,zhipuai_api_key)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()
