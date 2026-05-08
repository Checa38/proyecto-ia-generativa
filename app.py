import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List

# ─── Configuración ────────────────────────────────────────────────────────────
load_dotenv()

st.set_page_config(
    page_title="Agente Experto en Musculación",
    page_icon="https://www.gstatic.com/lamda/images/gemini_favicon_f069958c85030456e93de685481c559f160ea06.svg",
    layout="centered"
)

st.title("Asistente Experto en Musculación")
st.caption("Entrenador personal y nutricionista deportivo con IA  ·  Powered by Gemini + RAG")

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un entrenador personal y nutricionista deportivo experto, 
especializado en musculación y fitness. Tienes acceso a una base de conocimiento 
actualizada sobre entrenamiento, nutrición y técnica de ejercicios.

REGLAS:
- Responde SIEMPRE basándote primero en el contexto recuperado de la base de conocimiento.
- Si el contexto no contiene información suficiente, indícalo claramente y responde 
  con tu conocimiento general, señalándolo explícitamente.
- Usa un tono profesional pero cercano, como un entrenador real.
- Cuando des recomendaciones de entrenamiento, siempre menciona la importancia 
  del descanso y la recuperación.
- No recomiendas sustancias prohibidas ni tratamientos médicos.
- Si la pregunta no está relacionada con gimnasio o nutrición deportiva, 
  redirige amablemente al usuario hacia tu área de expertise.
"""

# ─── Estado del Grafo ─────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: List
    context: str

# ─── Inicialización (cacheada para no recargar en cada interacción) ───────────
@st.cache_resource
def init_agent():
    """Inicializa embeddings, vectorstore, LLM y grafo. Se ejecuta solo una vez."""
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = os.getenv("GEMINI_API_KEY")
    
    # Embeddings y vectorstore
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key,
        client_options={"api_endpoint": "generativelanguage.googleapis.com"},
        transport="rest"
    )
    
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
    )
    
    # Nodos del grafo
    def retrieve_node(state: AgentState) -> AgentState:
        last_message = state["messages"][-1].content
        docs = retriever.invoke(last_message)
        context = "\n\n".join([d.page_content for d in docs])
        return {**state, "context": context}

    def generate_node(state: AgentState) -> AgentState:
        context = state["context"]
        messages = state["messages"]
        system_with_context = f"""{SYSTEM_PROMPT}

CONTEXTO DE LA BASE DE CONOCIMIENTO:
{context}
"""
        full_messages = [SystemMessage(content=system_with_context)] + messages
        response = llm.invoke(full_messages)
        return {**state, "messages": messages + [response]}

    # Grafo
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    return graph

# ─── Inicializar historial de chat en session_state ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph_config" not in st.session_state:
    st.session_state.graph_config = {"configurable": {"thread_id": "streamlit_session"}}

# ─── Cargar agente ────────────────────────────────────────────────────────────
with st.spinner("Cargando base de conocimiento..."):
    graph = init_agent()

# ─── Mostrar historial de mensajes ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Input del usuario ────────────────────────────────────────────────────────
if user_input := st.chat_input("Pregunta sobre entrenamiento, nutrición o ejercicios..."):
    
    # Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Obtener respuesta del agente
    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conocimiento..."):
            
            config = st.session_state.graph_config
            state = graph.get_state(config)
            history = state.values.get("messages", []) if state.values else []
            
            new_state = {
                "messages": history + [HumanMessage(content=user_input)],
                "context": ""
            }
            
            result = graph.invoke(new_state, config=config)
            response = result["messages"][-1].content
        
        st.markdown(response)
    
    # Guardar respuesta en historial
    st.session_state.messages.append({"role": "assistant", "content": response})

# ─── Sidebar informativo ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Sobre este agente")
    st.markdown("""
    **Dominio:** Musculación y fitness
    
    **Puede ayudarte con:**
    - Nutrición deportiva y macros
    - Técnica de ejercicios
    - Programación del entrenamiento
    - Suplementación con evidencia
    - Recuperación y descanso
    
    **Stack tecnológico:**
    - LLM: Gemini 2.5 Flash
    - Embeddings: gemini-embedding-001
    - Vector DB: ChromaDB
    - Framework: LangGraph
    """)
    
    st.divider()
    
    if st.button("Limpiar conversación"):
        st.session_state.messages = []
        st.session_state.graph_config = {
            "configurable": {"thread_id": f"session_{os.urandom(4).hex()}"}
        }
        st.rerun()
    
    st.caption("Proyecto Final — IA Generativa")
