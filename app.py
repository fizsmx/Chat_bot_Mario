import chainlit as cl
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

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

@cl.on_chat_start
async def on_chat_start():
    if not api_key:
        await cl.Message(content="⚠️ **Error:** No se ha detectado la GOOGLE_API_KEY en el entorno.").send()
        return

    # Esperar a que el usuario suba archivos PDF
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="¡Bienvenido a **DataBrain IA**! 🧠\nPor favor, sube uno o más archivos **PDF** para comenzar el análisis interactivo.",
            accept=["application/pdf"],
            max_size_mb=50,
            timeout=300,
            max_files=10
        ).send()

    msg = cl.Message(content=f"⏳ Procesando {len(files)} archivo(s)...")
    await msg.send()

    # Paso 1: Extraer texto
    text = ""
    for file in files:
        pdf_reader = PdfReader(file.path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

    if not text.strip():
        msg.content = "❌ No se pudo extraer texto de los PDFs. Podrían estar escaneados o protegidos."
        await msg.update()
        return

    # Paso 2: Fragmentar
    msg.content = "⏳ Dividiendo y estructurando el contenido..."
    await msg.update()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    text_chunks = text_splitter.split_text(text)

    # Paso 3: Vectorizar localmente con FastEmbed
    msg.content = "⏳ Generando base de datos vectorial (Modelos Locales de Alta Velocidad)..."
    await msg.update()

    embeddings = FastEmbedEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    
    # Guardar la base de datos en la sesión del usuario
    cl.user_session.set("vector_store", vector_store)

    msg.content = "✅ **¡Documentos procesados con éxito!** Ya puedes hacerme preguntas sobre su contenido."
    await msg.update()

@cl.on_message
async def on_message(message: cl.Message):
    vector_store = cl.user_session.get("vector_store")
    if not vector_store:
        await cl.Message(content="⚠️ Por favor, sube y procesa un documento PDF primero recargando la página.").send()
        return

    # Buscar documentos similares
    docs = vector_store.similarity_search(message.content)
    chain = get_conversational_chain()
    
    # Mensaje temporal de "Pensando..."
    res_msg = cl.Message(content="")
    await res_msg.send()

    # Obtener respuesta
    res = chain.invoke({"input_documents": docs, "question": message.content})
    
    # Actualizar el mensaje con la respuesta final
    res_msg.content = res["output_text"]
    await res_msg.update()
