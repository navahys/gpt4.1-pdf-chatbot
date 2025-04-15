import os
import json
import glob
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

# Page configuration
st.set_page_config(
    page_title="PDF RAG with GPT-4.1",
    page_icon="📚",
    layout="wide"
)

# Constants
LOG_PATH = "chat_log.json"
MODEL_NAME = "gpt-4.1"
TEMP_DIR = "temp_pdfs"
DATA_DIR = "data"  # 기본 PDF 파일이 저장된 디렉토리
VECTOR_STORE_PATH = "vectorstore"  # 벡터 스토어 저장 경로

# Create required directories if they don't exist
for dir_path in [TEMP_DIR, DATA_DIR, VECTOR_STORE_PATH]:
    os.makedirs(dir_path, exist_ok=True)

# Helper functions
def load_api_key():
    """Load OpenAI API key from .env file or session state"""
    load_dotenv()
    
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        return st.session_state.openai_api_key
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다.")
    
    return api_key

def load_chat_history():
    """Load previous chat history from log file"""
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"채팅 기록을 로드하는 중 오류가 발생했습니다: {e}")
    return []

def save_chat_history(history):
    """Save chat history to log file"""
    try:
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"채팅 기록을 저장하는 중 오류가 발생했습니다: {e}")

def load_default_pdfs():
    """Load PDFs from the data directory"""
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        st.warning(f"'{DATA_DIR}' 디렉토리에 PDF 파일이 없습니다.")
        return None
    
    with st.spinner(f"{len(pdf_files)}개의 기본 PDF 파일을 처리하는 중입니다..."):
        all_docs = []
        processed_files = []
        
        for pdf_path in pdf_files:
            try:
                # Try with PyPDFLoader first
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # If empty, try with UnstructuredPDFLoader
                if not documents:
                    loader = UnstructuredPDFLoader(pdf_path)
                    documents = loader.load()
                
                all_docs.extend(documents)
                processed_files.append(os.path.basename(pdf_path))
                st.session_state.default_pdfs.append(os.path.basename(pdf_path))
            except Exception as e:
                st.error(f"'{os.path.basename(pdf_path)}' 처리 중 오류: {e}")
        
        if not all_docs:
            st.error("처리할 문서가 없습니다.")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # 법률 문서의 맥락을 더 잘 유지하기 위해 chunk_size 증가
            chunk_overlap=300,  # 더 많은 중복으로 법률 문맥 유지
            separators=["\n\n", "\n", ".", " ", ""],  # 법률 문서의 구조를 고려한 분할
        )
        texts = text_splitter.split_documents(all_docs)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=load_api_key())
        vectordb = FAISS.from_documents(texts, embeddings)
        
        # Save the vectorstore for future use
        vectordb.save_local(VECTOR_STORE_PATH)
        
        st.success(f"✅ {len(processed_files)}개의 기본 PDF 파일 처리 완료")
        return vectordb

def load_saved_vectorstore():
    """Load saved vector store if it exists"""
    if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH):
        try:
            embeddings = OpenAIEmbeddings(api_key=load_api_key())
            vectordb = FAISS.load_local(VECTOR_STORE_PATH, embeddings,allow_dangerous_deserialization=True)
            return vectordb
        except Exception as e:
            st.error(f"저장된 벡터 스토어를 로드하는 중 오류가 발생했습니다: {e}")
    return None

def process_pdfs(uploaded_files):
    """Process uploaded PDF files and merge with existing vector store"""
    with st.spinner("PDF 파일을 처리하는 중입니다..."):
        all_docs = []
        for uploaded_file in uploaded_files:
            # Save uploaded file to temp directory
            file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            try:
                # Try with PyPDFLoader first
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # If empty, try with UnstructuredPDFLoader
                if not documents:
                    loader = UnstructuredPDFLoader(file_path)
                    documents = loader.load()
                
                all_docs.extend(documents)
                st.session_state.processed_files.append(uploaded_file.name)
            except Exception as e:
                st.error(f"'{uploaded_file.name}' 처리 중 오류: {e}")
        
        if not all_docs:
            st.error("처리할 문서가 없습니다. 다른 PDF 파일을 업로드해주세요.")
            return st.session_state.vectordb  # Return existing vectordb
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )
        texts = text_splitter.split_documents(all_docs)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=load_api_key())
        
        # If we already have a vectordb, merge the new documents
        if st.session_state.vectordb:
            vectordb = st.session_state.vectordb
            vectordb.add_documents(texts)
        else:
            vectordb = FAISS.from_documents(texts, embeddings)
        
        # Save the updated vectorstore
        vectordb.save_local(VECTOR_STORE_PATH)
        
        return vectordb

def initialize_chain(vectordb, doc_weight=0.7, model_temperature=0.7):
    """Initialize the conversation chain with the vector store and balance settings"""
    api_key = load_api_key()
    if not api_key:
        return None
    
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        temperature=model_temperature,
        api_key=api_key
    )
    
    # Rewrite prompt for question improvement
    rewrite_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        당신은 산업재해 관련 법률 질문을 더 명확하고 구체적으로 개선하는 법률 전문 AI입니다.
        원본 질문: {question}
        
        법률적 맥락과 관련 조항을 찾기 위해 개선된 질문:
        """
    )   
    
    # QA prompt to include source documents with fixed doc_weight
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        당신은 산업재해 관련 법률 문서와 법적 지식을 결합하여 전문적인 법률 정보를 제공하는 AI 법률 조언자입니다.

        ## 법률 문서 내용:
        {{context}}

        ## 질문:
        {{question}}

        ## 답변 생성 지침:
        - 문서 내용 의존도: {doc_weight*9.5}/10 (10이면 문서 내용만 사용, 1이면 AI 지식 거의 전적으로 사용)
        - 산업안전보건법, 산업재해보상보험법 등 관련 법률에 기반하여 정확한, 법적으로 건전한 조언을 제공하세요
        - 법조항과 관련 판례가 있다면 명시적으로 언급하세요 (예: "산업안전보건법 제00조에 따르면...")
        - 사용자가 법적 절차, 권리, 의무에 대해 이해할 수 있도록 명확하게 설명하세요
        - 복잡한 법률 용어가 있다면 일반인도 이해할 수 있도록 풀어서 설명하세요
        - 문서에 특정 상황이 명시되지 않았다면 "귀하의 구체적인 상황에 따라 답변이 달라질 수 있으므로 전문 변호사와 상담을 권장합니다"라고 언급하세요
        - 답변 마지막에 항상 "이 정보는 일반적인 법률 정보 제공 목적이며, 개별 사례에 대한 법률 조언을 대체할 수 없습니다. 정확한 법률 자문을 위해서는 산재 전문 변호사와 상담하시기 바랍니다."라는 면책 문구를 포함하세요
        
        답변:
        """
    )
    
    # Set up memory and conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Improved chain with source documents
    rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 7}),
        memory=memory,
        condense_question_prompt=rewrite_prompt,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
    )
    
    st.session_state.rewrite_chain = rewrite_chain
    return conversation_chain

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "rewrite_chain" not in st.session_state:
    st.session_state.rewrite_chain = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "default_pdfs" not in st.session_state:
    st.session_state.default_pdfs = []

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

if "doc_weight" not in st.session_state:
    st.session_state.doc_weight = 7

if "model_temperature" not in st.session_state:
    st.session_state.model_temperature = 0.7

if "vectordb" not in st.session_state:
    # Try to load existing vectorstore first
    st.session_state.vectordb = load_saved_vectorstore()
    
    # If no existing vectorstore, load default PDFs
    if st.session_state.vectordb is None:
        st.session_state.vectordb = load_default_pdfs()
        
    # Initialize conversation chain if vectordb exists
    if st.session_state.vectordb:
        st.session_state.conversation_chain = initialize_chain(
            st.session_state.vectordb,
            st.session_state.doc_weight / 10,
            st.session_state.model_temperature
        )

# UI Layout
st.title("📄 PDF 기반 GPT-4.1 챗봇")
st.markdown("사전 로드된 PDF 파일을 기반으로 대화하거나, 추가 PDF 파일을 업로드하세요.")

# Sidebar for API Key and settings
with st.sidebar:
    st.header("설정")
    
    # API Key input
    api_key_input = st.text_input(
        "OpenAI API 키",
        type="password",
        value=st.session_state.openai_api_key,
        help="OpenAI API 키를 입력하세요. .env 파일에 설정할 수도 있습니다."
    )
    
    if api_key_input:
        st.session_state.openai_api_key = api_key_input
    
    # Show loaded default files
    if st.session_state.default_pdfs:
        st.subheader("기본 로드된 PDF 파일")
        for file in st.session_state.default_pdfs:
            st.write(f"- {file}")
    
    # Additional file upload
    st.subheader("추가 PDF 파일 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드 (여러 개 가능)",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("📄 추가 PDF 처리하기"):
        st.session_state.vectordb = process_pdfs(uploaded_files)
        if st.session_state.vectordb:
            st.session_state.conversation_chain = initialize_chain(
                st.session_state.vectordb, 
                st.session_state.doc_weight / 10,
                st.session_state.model_temperature
            )
            st.success(f"✅ {len(uploaded_files)}개 추가 PDF 파일 처리 완료")
    
    # Show processed user files
    if st.session_state.processed_files:
        st.subheader("사용자 추가 PDF 파일")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    # 데이터 관리 옵션
    st.divider()
    st.subheader("데이터 관리")
    
    if st.button("🔄 기본 PDF 재처리"):
        st.session_state.default_pdfs = []
        st.session_state.vectordb = load_default_pdfs()
        if st.session_state.vectordb:
            st.session_state.conversation_chain = initialize_chain(
                st.session_state.vectordb, 
                st.session_state.doc_weight / 10,
                st.session_state.model_temperature
            )
    
    if st.button("🗑️ 벡터 스토어 초기화", type="primary"):
        if os.path.exists(VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        st.session_state.vectordb = None
        st.session_state.conversation_chain = None
        st.session_state.default_pdfs = []
        st.session_state.processed_files = []
        st.success("✅ 벡터 스토어가 초기화되었습니다. 페이지를 새로고침하세요.")
    
    # Response balance settings
    st.divider()
    st.subheader("응답 설정")
    
    # Document weight slider
    doc_weight = st.slider(
        "문서 의존도",
        min_value=1,
        max_value=10,
        value=st.session_state.doc_weight,
        help="10에 가까울수록 문서 내용에 충실하게, 1에 가까울수록 AI의 지식을 더 활용합니다."
    )
    
    # Model temperature slider
    model_temp = st.slider(
        "창의성 수준",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.model_temperature,
        step=0.1,
        help="높을수록 더 창의적인 답변, 낮을수록 더 일관된 답변을 생성합니다."
    )
    
    # Apply settings button
    if st.button("설정 적용"):
        if doc_weight != st.session_state.doc_weight or model_temp != st.session_state.model_temperature:
            st.session_state.doc_weight = doc_weight
            st.session_state.model_temperature = model_temp
            
            # Reinitialize chain with new settings if vectordb exists
            if st.session_state.vectordb:
                st.session_state.conversation_chain = initialize_chain(
                    st.session_state.vectordb,
                    doc_weight / 10,
                    model_temp
                )
                st.success("✅ 새로운 설정이 적용되었습니다.")
    
    # Settings info
    current_settings = f"""
    **현재 설정:**
    - 문서 의존도: {st.session_state.doc_weight}/10
    - 창의성 수준: {st.session_state.model_temperature:.1f}
    """
    st.markdown(current_settings)
    
    # History management
    st.divider()
    st.subheader("대화 기록 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 대화 저장"):
            save_chat_history(st.session_state.chat_history)
            st.success("✅ 대화 저장 완료")
    
    with col2:
        if st.button("📂 대화 불러오기"):
            st.session_state.chat_history = load_chat_history()
            st.success("✅ 이전 대화 불러오기 완료")
    
    if st.button("🗑️ 대화 초기화"):
        st.session_state.chat_history = []
        st.success("✅ 대화 내역을 초기화했습니다.")

# Main chat area
if not st.session_state.vectordb:
    st.warning("⚠️ 'data' 디렉토리에 기본 PDF 파일이 없거나 처리 중 오류가 발생했습니다. PDF 파일을 업로드하여 시작하세요.")
elif not st.session_state.conversation_chain:
    st.info("👈 사이드바에서 'PDF 처리하기' 버튼을 클릭하여 시작하세요.")
else:
    st.success(f"✅ {len(st.session_state.default_pdfs) + len(st.session_state.processed_files)}개의 PDF 파일이 로드되었습니다. 대화를 시작하세요!")

# Display chat history
chat_container = st.container()
with chat_container:
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)

# User input
if st.session_state.conversation_chain:
    user_question = st.chat_input("질문을 입력하세요")
    
    if user_question:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Process the question and get a response
        with st.spinner("답변을 생성하는 중..."):
            try:
                # Improve the question
                improved_question = st.session_state.rewrite_chain.run(user_question)
                
                # Get response with improved question and current document weight
                response = st.session_state.conversation_chain.run(improved_question)
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Add to history
                st.session_state.chat_history.append((user_question, response))
                
                # Auto-save history
                save_chat_history(st.session_state.chat_history)
                
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.error("상세 정보: " + str(type(e).__name__) + " - " + str(e))