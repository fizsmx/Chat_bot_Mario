import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar la API de Google
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, 
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Responde a la pregunta de la manera más detallada posible basándote en el contexto proporcionado. 
    Si la respuesta no se encuentra en el contexto proporcionado, simplemente di "No puedo encontrar la respuesta en el documento". No inventes la respuesta.
    
    Contexto:
    {context}
    
    Pregunta:
    {question}
    
    Respuesta:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    
    # Cargar la base de datos de FAISS guardada
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error("Por favor, procesa primero un archivo PDF.")
        return

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    
    st.write("🤖 **Respuesta:**", response["output_text"])

def main():
    st.set_page_config(page_title="AI Core Matrix", page_icon="💻", layout="wide")
    
    # CSS Futurista (Neon, Dark Mode, Matrix Style)
    st.markdown("""
    <style>
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at center, #0a1c10 0%, #050505 100%);
        color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
    }
    h1 {
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 10px #00ff41, 0 0 20px #00ff41;
        color: #00ff41 !important;
        padding-bottom: 20px;
    }
    /* Estilos de botones */
    div.stButton > button {
        background-color: transparent !important;
        color: #00ff41 !important;
        border: 2px solid #00ff41 !important;
        border-radius: 8px !important;
        box-shadow: 0 0 10px #00ff41 inset, 0 0 10px #00ff41 !important;
        text-transform: uppercase;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #00ff41 !important;
        color: black !important;
        box-shadow: 0 0 20px #00ff41 inset, 0 0 30px #00ff41 !important;
    }
    /* Estilos del input */
    .stTextInput>div>div>input {
        background-color: #111 !important;
        color: #00ff41 !important;
        border: 1px solid #00ff41 !important;
        box-shadow: 0 0 5px #00ff41 !important;
    }
    /* Estilos del sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #00ff41;
        box-shadow: 2px 0 15px rgba(0,255,65,0.2);
    }
    /* Color de textos secundarios */
    .stMarkdown, p, div, span {
        color: #00ff41 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.header("⚡ SISTEMA DE ANÁLISIS IA - PDF MATRIX ⚡")

    if not api_key:
        st.warning("⚠️ No se ha detectado la GOOGLE_API_KEY. Por favor, asegúrate de configurarla en tu archivo .env o en las variables de entorno.")

    user_question = st.text_input("Haz una pregunta sobre los documentos PDF que has subido:")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📂 Menú:")
        pdf_docs = st.file_uploader("Sube tus archivos PDF y haz clic en 'Procesar'", accept_multiple_files=True, type=["pdf"])
        if st.button("Procesar"):
            if not pdf_docs:
                st.error("Por favor, sube al menos un documento PDF primero.")
            elif not api_key:
                st.error("Por favor, configura tu API Key de Google antes de procesar.")
            else:
                with st.spinner("Procesando documentos..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip() == "":
                        st.error("No se pudo extraer texto de los PDFs. Podrían estar escaneados o protegidos.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("¡Documentos procesados correctamente! Ya puedes hacer preguntas.")

if __name__ == "__main__":
    main()
