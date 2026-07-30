import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Configuración inicial de la página
st.set_page_config(page_title="AI Document Analyst", page_icon="📑", layout="wide")

# Lógica del Tema (Claro / Oscuro)
if "theme" not in st.session_state:
    st.session_state.theme = "dark" # Por defecto oscuro para que se vea futurista pero profesional

def apply_theme():
    if st.session_state.theme == "dark":
        st.markdown("""
        <style>
        .stApp { background-color: #0E1117 !important; color: #E0E6ED !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stChatMessage { background-color: #1A1C23 !important; border: 1px solid #2D3139 !important; border-radius: 12px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .stChatMessage p, .stMarkdown p, h1, h2, h3, h4, h5, h6, span { color: #E0E6ED !important; }
        section[data-testid="stSidebar"] { background-color: #14161B !important; border-right: 1px solid #2D3139; }
        .stTextInput>div>div>input { background-color: #1A1C23 !important; color: #E0E6ED !important; border: 1px solid #3B4252 !important; border-radius: 8px; }
        div.stButton > button { background-color: #2962FF !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: bold; transition: all 0.3s ease; }
        div.stButton > button:hover { background-color: #1E4BD8 !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(41,98,255,0.4) !important; }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp { background-color: #F5F7FA !important; color: #2C3E50 !important; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .stChatMessage { background-color: #FFFFFF !important; border: 1px solid #E4E7EB !important; border-radius: 12px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
        .stChatMessage p, .stMarkdown p, h1, h2, h3, h4, h5, h6, span { color: #2C3E50 !important; }
        section[data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E4E7EB; }
        .stTextInput>div>div>input { background-color: #FFFFFF !important; color: #2C3E50 !important; border: 1px solid #CBD5E0 !important; border-radius: 8px; }
        div.stButton > button { background-color: #2962FF !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: bold; transition: all 0.3s ease; }
        div.stButton > button:hover { background-color: #1E4BD8 !important; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(41,98,255,0.3) !important; }
        </style>
        """, unsafe_allow_html=True)

# Aplicar el CSS del tema seleccionado
apply_theme()

# Cargar variables de entorno
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_conversational_chain():
    prompt_template = """
    Eres un analista de documentos IA avanzado y profesional. Responde a la pregunta de la manera más clara, detallada y estructurada posible basándote estrictamente en el contexto proporcionado. 
    Si la respuesta no se encuentra en el contexto proporcionado, indica amablemente que el documento no contiene esa información. No inventes respuestas.
    
    Contexto extraído del documento:
    {context}
    
    Pregunta del usuario:
    {question}
    
    Análisis y Respuesta:
    """
    model = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def process_user_question(user_question):
    embeddings = FastEmbedEmbeddings()
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return "⚠️ Por favor, procesa primero un archivo PDF en el panel lateral antes de hacer preguntas."

    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    # Botón para cambiar el tema en la parte superior derecha
    col1, col2 = st.columns([8, 1])
    with col1:
        st.title("📑 DataBrain - Análisis de PDFs con IA")
        st.markdown("Sube tus documentos PDF en el panel lateral y utiliza este chat interactivo para extraer información, resúmenes y respuestas precisas al instante.")
    with col2:
        if st.button("🌓 Tema"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

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
            with st.spinner("Analizando información..."):
                response = process_user_question(prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Panel lateral profesional
    with st.sidebar:
        st.title("📂 Gestión de Documentos")
        st.markdown("---")
        pdf_docs = st.file_uploader("Selecciona tus archivos PDF", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Procesar Documentos", use_container_width=True):
            if not pdf_docs:
                st.warning("Selecciona al menos un documento PDF.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Paso 1: Leer
                status_text.markdown("⏳ **Paso 1/3:** Extrayendo texto de los PDFs...")
                text = ""
                total_pages = sum(len(PdfReader(pdf).pages) for pdf in pdf_docs)
                current_page = 0
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        text += page.extract_text() or ""
                        current_page += 1
                        # Actualizar barra hasta el 40%
                        progress_bar.progress(int((current_page / total_pages) * 40))
                
                if not text.strip():
                    st.error("No se pudo extraer texto. Verifica que los PDFs no sean solo imágenes escaneadas.")
                else:
                    # Paso 2: Fragmentar
                    status_text.markdown("⏳ **Paso 2/3:** Dividiendo y estructurando el contenido...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
                    text_chunks = text_splitter.split_text(text)
                    progress_bar.progress(60)
                    
                    # Paso 3: Vectorizar
                    status_text.markdown("⏳ **Paso 3/3:** Generando base de datos vectorial (Modelos locales)...")
                    embeddings = FastEmbedEmbeddings()
                    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                    vector_store.save_local("faiss_index")
                    progress_bar.progress(100)
                    status_text.markdown("✅ **¡Documentos procesados con éxito!** El asistente está listo.")
        
        st.markdown("---")
        st.caption("Desarrollado con Streamlit & Google Gemini AI")

if __name__ == "__main__":
    main()
