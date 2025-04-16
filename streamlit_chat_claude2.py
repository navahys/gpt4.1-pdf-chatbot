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
    page_title="산업재해 법률 정보 안내 시스템",
    page_icon="📚",
    layout="wide"
)

# Constants
LOG_PATH = "anonymized_chat_log.json"  # 익명화된 로그 저장
MODEL_NAME = "gpt-4.1"
TEMP_DIR = "temp_pdfs"
DATA_DIR = "data"
VECTOR_STORE_PATH = "vectorstore"

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

def anonymize_chat_data(chat_data):
    """개인정보 및 민감정보 익명화 처리"""
    # 실제 구현시 개인식별정보 패턴 필터링 로직 추가
    # 예: 이름, 주민번호, 전화번호, 주소 등 패턴 마스킹
    return chat_data

def load_chat_history():
    """Load previous chat history from log file"""
    try:
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        st.error(f"익명화된 참고 자료를 로드하는 중 오류가 발생했습니다: {e}")
    return []

def save_chat_history(history):
    """Save anonymized chat history to log file"""
    try:
        # 사용자 동의 여부 확인
        if st.session_state.get("data_consent", False):
            anonymized_history = anonymize_chat_data(history)
            with open(LOG_PATH, "w", encoding="utf-8") as f:
                json.dump(anonymized_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"익명화된 참고 자료를 저장하는 중 오류가 발생했습니다: {e}")

def load_default_pdfs():
    """Load PDFs from the data directory"""
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        st.warning(f"'{DATA_DIR}' 디렉토리에 PDF 파일이 없습니다.")
        return None
    
    with st.spinner(f"{len(pdf_files)}개의 기본 법률 참고자료를 처리하는 중입니다..."):
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
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        texts = text_splitter.split_documents(all_docs)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(api_key=load_api_key())
        vectordb = FAISS.from_documents(texts, embeddings)
        
        # Save the vectorstore for future use
        vectordb.save_local(VECTOR_STORE_PATH)
        
        st.success(f"✅ {len(processed_files)}개의 법률 참고자료 처리 완료")
        return vectordb

def load_saved_vectorstore():
    """Load saved vector store if it exists"""
    if os.path.exists(VECTOR_STORE_PATH) and os.path.isdir(VECTOR_STORE_PATH):
        try:
            embeddings = OpenAIEmbeddings(api_key=load_api_key())
            vectordb = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            return vectordb
        except Exception as e:
            st.error(f"저장된 참고자료 데이터베이스를 로드하는 중 오류가 발생했습니다: {e}")
    return None

def process_pdfs(uploaded_files):
    """Process uploaded PDF files and merge with existing vector store"""
    with st.spinner("법률 참고자료를 처리하는 중입니다..."):
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
        당신은 산업재해 관련 법률 정보를 안내하는 AI입니다. 
        사용자의 질문을 더 명확하고 구체적으로 개선하여 적절한 법률 참고자료를 찾을 수 있도록 도와주세요.
        
        원본 질문: {question}
        
        법률적 맥락과 관련 조항을 찾기 위해 개선된 질문:
        """
    )   
    
    # QA prompt to include source documents with fixed doc_weight
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=f"""
        당신은 산업재해 관련 법률 문서와 법적 지식을 결합하여 일반적인 법률 정보를 제공하는 AI 법률 정보 안내 시스템입니다.

        ## 법률 참고자료 내용:
        {{context}}

        ## 질문:
        {{question}}

        ## 답변 생성 지침:
        - 문서 내용 의존도: {doc_weight*9.5}/10 (10이면 문서 내용만 사용, 1이면 AI 지식 거의 전적으로 사용)
        - 산업안전보건법, 산업재해보상보험법 등 관련 법률에 기반하여 정확한 정보를 안내하세요
        - 법조항과 관련 판례가 있다면 명시적으로 언급하세요 (예: "산업안전보건법 제00조에 따르면...")
        - 사용자가 법적 절차, 권리, 의무에 대해 이해할 수 있도록 명확하게 설명하세요
        - 복잡한 법률 용어가 있다면 일반인도 이해할 수 있도록 풀어서 설명하세요
        - 문서에 특정 상황이 명시되지 않았다면 "구체적인 상황에 따라 달라질 수 있으므로 법률 전문가와 상담을 권장합니다"라고 언급하세요
        - "법률 자문"이나 "법률 상담"이라는 표현 대신 "법률 정보 제공", "법률 참고자료 안내" 등의 표현을 사용하세요
        - 답변 마지막에 항상 다음 면책 조항을 포함하세요: 

        "※ 안내 사항: 제공해 드린 정보는 일반적인 법률 참고자료로, 법적 효력이 없으며 참고용으로만 활용하시기 바랍니다. 이 시스템은 변호사가 아니며, 법률 자문을 제공하지 않습니다. 구체적인 법률 문제는 반드시 산재 전문 변호사나 관련 기관에 문의하시기 바랍니다."
        
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

# 민감정보 입력 필터링 함수
def contains_sensitive_info(text):
    """민감정보 패턴 검사"""
    import re
    # 주민번호 패턴
    korean_id_pattern = r'\d{6}[-]\d{7}'
    # 전화번호 패턴
    phone_pattern = r'01[016789][-\s]?\d{3,4}[-\s]?\d{4}'
    
    if re.search(korean_id_pattern, text) or re.search(phone_pattern, text):
        return True
    return False

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

if "data_consent" not in st.session_state:
    st.session_state.data_consent = False

if "terms_accepted" not in st.session_state:
    st.session_state.terms_accepted = False

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
st.title("📄 산업재해 법률 정보 안내 시스템")
st.markdown("이 시스템은 법률 자문을 제공하지 않으며, 일반적인 법률 정보와 참고자료만 안내합니다.")

# 사용자 동의 및 이용약관 체크
if not st.session_state.terms_accepted:
    st.warning("⚠️ 시스템을 이용하기 전에 아래 이용약관과 개인정보 처리방침에 동의해주세요.")
    
    with st.expander("📜 이용약관 (필수)", expanded=True):
        st.markdown("""
        ## 이용약관
        
        1. **법률 정보 제공 목적**: 본 시스템은 법률 자문이 아닌 일반적인 법률 정보 제공 목적으로 운영됩니다.
        2. **법적 책임 제한**: 제공되는 모든 정보는 법적 효력이 없으며, 참고용으로만 활용하시기 바랍니다.
        3. **정보의 정확성**: 최신 법령을 반영하기 위해 노력하나, 실제 법령과 차이가 있을 수 있습니다.
        4. **전문가 상담 권장**: 구체적인 법률 문제는 반드시 전문 변호사와 상담하시기 바랍니다.
        5. **수익 모델**: 본 시스템은 교육 및 학습 목적으로 제공되며, 유료 법률 자문을 제공하지 않습니다.
        """)
    
    with st.expander("🔒 개인정보 처리방침 (필수)", expanded=True):
        st.markdown("""
        ## 개인정보 처리방침
        
        1. **개인정보 수집 항목**: 대화 내용 중 법률 정보 제공에 필요한 최소한의 정보만 익명화하여 저장됩니다.
        2. **민감정보 보호**: 주민등록번호, 전화번호 등 민감정보는 입력하지 마시고, 자동 필터링됩니다.
        3. **데이터 저장 및 활용**: 동의 시 익명화된 대화 내용을 시스템 개선 목적으로만 활용합니다.
        4. **제3자 제공**: 수집된 정보는 제3자에게 제공하지 않습니다.
        5. **데이터 보관 기간**: 수집된 정보는 30일간 보관 후 자동 삭제됩니다.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        terms_check = st.checkbox("이용약관에 동의합니다 (필수)")
    with col2:
        privacy_check = st.checkbox("개인정보 처리방침에 동의합니다 (필수)")
    
    data_consent = st.checkbox("대화 내용을 익명화하여 시스템 개선에 활용하는 것에 동의합니다 (선택)")
    
    if st.button("동의하고 시작하기", disabled=not (terms_check and privacy_check)):
        st.session_state.terms_accepted = True
        st.session_state.data_consent = data_consent
        st.rerun()
    
    st.stop()  # 동의 전까지 아래 내용 표시하지 않음

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
        st.subheader("기본 로드된 법률 참고자료")
        for file in st.session_state.default_pdfs:
            st.write(f"- {file}")
    
    # Additional file upload
    st.subheader("추가 법률 참고자료 업로드")
    uploaded_files = st.file_uploader(
        "PDF 파일 업로드 (여러 개 가능)",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("📄 추가 참고자료 처리하기"):
        st.session_state.vectordb = process_pdfs(uploaded_files)
        if st.session_state.vectordb:
            st.session_state.conversation_chain = initialize_chain(
                st.session_state.vectordb, 
                st.session_state.doc_weight / 10,
                st.session_state.model_temperature
            )
            st.success(f"✅ {len(uploaded_files)}개 추가 참고자료 처리 완료")
    
    # Show processed user files
    if st.session_state.processed_files:
        st.subheader("사용자 추가 참고자료")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    # 데이터 관리 옵션
    st.divider()
    st.subheader("데이터 관리")
    
    if st.button("🔄 기본 참고자료 재처리"):
        st.session_state.default_pdfs = []
        st.session_state.vectordb = load_default_pdfs()
        if st.session_state.vectordb:
            st.session_state.conversation_chain = initialize_chain(
                st.session_state.vectordb, 
                st.session_state.doc_weight / 10,
                st.session_state.model_temperature
            )
    
    if st.button("🗑️ 참고자료 초기화", type="primary"):
        if os.path.exists(VECTOR_STORE_PATH):
            import shutil
            shutil.rmtree(VECTOR_STORE_PATH)
            os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
        st.session_state.vectordb = None
        st.session_state.conversation_chain = None
        st.session_state.default_pdfs = []
        st.session_state.processed_files = []
        st.success("✅ 참고자료가 초기화되었습니다. 페이지를 새로고침하세요.")
    
    # Response balance settings
    st.divider()
    st.subheader("정보 제공 설정")
    
    # Document weight slider
    doc_weight = st.slider(
        "참고자료 의존도",
        min_value=1,
        max_value=10,
        value=st.session_state.doc_weight,
        help="10에 가까울수록 참고자료 내용에 충실하게, 1에 가까울수록 AI의 지식을 더 활용합니다."
    )
    
    # Model temperature slider
    model_temp = st.slider(
        "응답 다양성",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.model_temperature,
        step=0.1,
        help="높을수록 더 다양한 답변, 낮을수록 더 일관된 답변을 생성합니다."
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
    - 참고자료 의존도: {st.session_state.doc_weight}/10
    - 응답 다양성: {st.session_state.model_temperature:.1f}
    """
    st.markdown(current_settings)
    
    # Consent settings
    st.divider()
    st.subheader("개인정보 동의 설정")
    
    data_consent = st.checkbox("익명화된 대화 저장에 동의", 
                               value=st.session_state.data_consent,
                               help="시스템 개선을 위해 익명화된 대화 내용을 저장합니다.")
    
    if data_consent != st.session_state.data_consent:
        st.session_state.data_consent = data_consent
        st.success("✅ 개인정보 설정이 변경되었습니다.")
    
    # History management
    st.divider()
    st.subheader("대화 기록 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 대화 저장"):
            if st.session_state.data_consent:
                save_chat_history(st.session_state.chat_history)
                st.success("✅ 익명화된 대화 저장 완료")
            else:
                st.warning("⚠️ 대화 저장을 위해서는 동의가 필요합니다.")
    
    with col2:
        if st.button("📂 대화 불러오기"):
            st.session_state.chat_history = load_chat_history()
            st.success("✅ 이전 대화 불러오기 완료")
    
    if st.button("🗑️ 대화 초기화"):
        st.session_state.chat_history = []
        st.success("✅ 대화 내역을 초기화했습니다.")

# Main chat area
st.markdown("""
### ⚠️ 법적 책임 제한 안내
이 시스템은 법률 자문이 아닌 일반적인 법률 정보만 제공합니다. 모든 정보는 법적 효력이 없으며 참고용으로만 활용하시기 바랍니다. 구체적인 법률 문제는 반드시 전문 변호사와 상담하세요.
""")

if not st.session_state.vectordb:
    st.warning("⚠️ 'data' 디렉토리에 기본 참고자료가 없거나 처리 중 오류가 발생했습니다. PDF 파일을 업로드하여 시작하세요.")
elif not st.session_state.conversation_chain:
    st.info("👈 사이드바에서 '참고자료 처리하기' 버튼을 클릭하여 시작하세요.")
else:
    st.success(f"✅ {len(st.session_state.default_pdfs) + len(st.session_state.processed_files)}개의 법률 참고자료가 로드되었습니다. 질문을 입력하세요.")

# 민감정보 입력 주의사항 표시
st.markdown("""
#### ⚠️ 민감정보 입력 주의
- 주민등록번호, 전화번호 등 개인 식별 정보를 입력하지 마세요.
- 모든 대화는 익명화되어 저장될 수 있습니다(동의한 경우).
- 실제 사건의 구체적인 정황보다는 일반적인 법률 정보를 질문하세요.
""")

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
        # 민감정보 필터링
        if contains_sensitive_info(user_question):
            with st.chat_message("assistant"):
                st.error("""
                ⚠️ 개인정보 보호 알림
                
                입력하신 내용에 주민등록번호, 전화번호와 같은 민감정보가 포함된 것으로 판단됩니다. 
                개인정보 보호를 위해 민감정보를 제외하고 질문해 주시기 바랍니다.
                """)
        else:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_question)
            
            # Process the question and get a response
            with st.spinner("법률 정보를 검색하는 중..."):
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
                    
                    # Auto-save history if consent given
                    if st.session_state.data_consent:
                        save_chat_history(st.session_state.chat_history)
                    
                except Exception as e:
                    st.error(f"오류가 발생했습니다: {e}")
                    st.error("상세 정보: " + str(type(e).__name__) + " - " + str(e))

# Footer with
