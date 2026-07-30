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
    
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
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
    st.set_page_config(page_title="Chat con PDFs", page_icon="📄")
    st.header("Chatbot para leer y responder preguntas de PDFs 📄🤖")
    
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
