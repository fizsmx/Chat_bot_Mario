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

# Configuración inicial de la página
st.set_page_config(page_title="AI Document Analyst", page_icon="📑", layout="wide")

# CSS Profesional y Moderno
st.markdown("""
<style>
    /* Fondo principal y fuentes */
    .stApp {
        background-color: #f8f9fa;
        color: #2c3e50;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Encabezado principal */
    h1, h2, h3 {
        color: #1a2b3c !important;
        font-weight: 700 !important;
    }
    
    /* Estilos del Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e9ecef;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    /* Botones profesionales */
    div.stButton > button {
        background-color: #3498db !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(52, 152, 219, 0.2) !important;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #2980b9 !important;
        transform: translateY(-1px);
        box-shadow: 0 6px 8px rgba(52, 152, 219, 0.3) !important;
    }
    
    /* Mensajes de Chat */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Cargar variables de entorno
load_dotenv()
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Eres un analista de documentos profesional. Responde a la pregunta de la manera más clara, detallada y estructurada posible basándote estrictamente en el contexto proporcionado. 
    Si la respuesta no se encuentra en el contexto proporcionado, indica amablemente que el documento no contiene esa información. No inventes respuestas.
    
    Contexto:
    {context}
    
    Pregunta:
    {question}
    
    Respuesta Profesional:
    """
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return "⚠️ Por favor, procesa primero un archivo PDF en el panel lateral antes de hacer preguntas."

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.title("📑 Asistente de Análisis de Documentos IA")
    st.markdown("Sube tus documentos PDF en el panel lateral y utiliza este chat interactivo para extraer información, resúmenes y respuestas precisas.")

    if not api_key:
        st.error("⚠️ No se ha detectado la GOOGLE_API_KEY. Por favor, configúrala en tu entorno.")

    # Inicializar el historial de chat en el estado de la sesión
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada de chat interactiva
    if prompt := st.chat_input("Escribe tu pregunta sobre el documento aquí..."):
        # Añadir mensaje del usuario al historial y mostrarlo
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generar y mostrar la respuesta del asistente
        with st.chat_message("assistant"):
            with st.spinner("Analizando documentos..."):
                response = process_user_question(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Panel lateral profesional
    with st.sidebar:
        st.title("Gestión de Documentos")
        st.markdown("---")
        pdf_docs = st.file_uploader("Selecciona tus archivos PDF", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Procesar Documentos"):
            if not pdf_docs:
                st.warning("Selecciona al menos un documento PDF.")
            else:
                with st.spinner("Extrayendo y vectorizando texto..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No se pudo extraer texto. Verifica que los PDFs no sean solo imágenes escaneadas.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("✅ Documentos procesados con éxito. El asistente está listo.")
        st.markdown("---")
        st.caption("Desarrollado con Streamlit & Google Gemini AI")

if __name__ == "__main__":
    main()
