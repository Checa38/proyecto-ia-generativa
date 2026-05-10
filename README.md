# Agente Experto en Musculación y Fitness

---

## 1. Descripción Breve

El gran desarrollo de los modelos de lenguaje de gran escala ha abierto la posibilidad de construir asistentes especializados capaces de responder preguntas complejas sobre un dominio concreto. Con este proyecto se pretende aprovechar esta capacidad para crear un agente experto en musculación y fitness que, en lugar de depender únicamente del conocimiento general del modelo, consulte una base de conocimiento vectorial propia antes de responder.

Para ello se ha implementado un pipeline de Retrieval-Augmented Generation (RAG) usando Google Gemini como modelo de lenguaje y sistema de embeddings, ChromaDB como base de datos vectorial y LangGraph como framework del agente. Además, el agente mantiene memoria de conversación entre turnos, lo que le permite recordar el contexto de preguntas anteriores y mantener coherencia a lo largo de la sesión.

La aplicación puede usarse tanto desde el propio notebook como desde una interfaz web desplegada en **Streamlit Cloud**: 

[![Streamlit App](https://img.shields.io/badge/Streamlit-Open%20App-FF4B4B?style=for-the-badge&logo=streamlit)](https://proyecto-ia-generativa-3kbcelwu9ukkrzvehgpybo.streamlit.app/)
---

## 2. Dominio Elegido

El agente está especializado en **musculación y fitness**. Concretamente, la base de conocimiento cubre los siguientes temas:

- **Técnica de ejercicios**: descripción y puntos clave de los ejercicios de fuerza más utilizados, incluyendo sentadilla, peso muerto, press de banca, dominadas y remo, entre otros
- **Entrenamiento e hipertrofia**: principios de sobrecarga progresiva, frecuencia, volumen semanal por grupo muscular y estructuras de rutina como Full Body, Push-Pull-Legs o Torso-Pierna
- **Nutrición deportiva**: cálculo de calorías y macronutrientes, timing nutricional pre y post entrenamiento, y suplementación con evidencia científica
- **Programación**: periodización lineal y ondulante, semanas de descarga y progresión a largo plazo
- **Recuperación**: importancia del sueño, señales de sobreentrenamiento y gestión del descanso entre sesiones

---

## 3. Stack Tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | `models/gemini-embedding-001` |
| Base de conocimiento vectorial | ChromaDB (persistida en `./chroma_db`) |
| Framework del agente | LangGraph + LangChain |
| Memoria de conversación | `MemorySaver` de LangGraph |
| Carga de documentos | `PyPDFLoader` de LangChain Community |
| Interfaz web | Streamlit |
| Entorno de desarrollo | Jupyter Notebook / VS Code |

---

## 4. Guía de Ejecución

### Requisitos previos

- Python 3.10+
- API Key de Google Gemini, obtenible en https://aistudio.google.com/app/apikey

### Instalación de dependencias

```bash
pip install langchain langchain-google-genai langchain-chroma langchain-community \
            langchain-text-splitters langgraph chromadb pypdf python-dotenv \
            google-generativeai streamlit
```

### Configuración de la API Key

Crear un archivo `.env` en la raíz del proyecto con el siguiente contenido:

```
GEMINI_API_KEY=tu_clave_aqui
```

### Ejecución del notebook

1. Colocar los 3 PDFs en la carpeta `data/`:
   - `els50exercicisdeforsamesutilitzats.pdf`
   - `ENCICLOPEDIA-DE-EJERCICIOS-ACTUALIZADO.pdf`
   - `Libbys-Guia-Nutricion-Deportiva.pdf`

2. Abrir `notebook.ipynb` en Jupyter o VS Code

3. Ejecutar todas las celdas en orden con `Kernel → Restart & Run All`

4. Interactuar con el agente desde la celda de chat al final del notebook

> La primera ejecución indexa los documentos en ChromaDB y puede tardar 2-3 minutos debido a los límites de la API gratuita. Las ejecuciones siguientes cargan el índice ya existente directamente.

### Ejecución de la interfaz web (Streamlit)

```bash
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`. La app también está desplegada públicamente en:

[![Streamlit App](https://img.shields.io/badge/Streamlit-Open%20App-FF4B4B?style=for-the-badge&logo=streamlit)](https://proyecto-ia-generativa-3kbcelwu9ukkrzvehgpybo.streamlit.app/)

---

## 5. Justificación del System Prompt

```
Eres un entrenador personal y nutricionista deportivo experto, 
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
```

El diseño del system prompt parte de una idea fundamental: sin una definición clara del rol, un modelo de lenguaje general tenderá a dar respuestas genéricas que no aprovechan la base de conocimiento. Por ello, la primera decisión fue establecer explícitamente que el agente es un entrenador personal y nutricionista deportivo experto, lo que condiciona el tono, el vocabulario y el nivel de profundidad de todas las respuestas.

La regla más importante del prompt es que el agente debe basarse siempre primero en el contexto recuperado por el sistema RAG. Esto garantiza que ChromaDB sea el motor real de las respuestas y no un elemento decorativo. Cuando el contexto no contiene información suficiente, el agente lo indica explícitamente, lo que permite al usuario distinguir cuándo la respuesta proviene de los documentos y cuándo del conocimiento general del modelo.

Se ha optado por un tono profesional pero cercano, similar al de un entrenador real, ya que un agente de fitness excesivamente técnico o clínico resultaría poco útil en la práctica. Además, se ha incluido como regla obligatoria mencionar la importancia del descanso y la recuperación en cualquier recomendación de entrenamiento, pues es un aspecto frecuentemente olvidado pero tan relevante como el propio entreno.

Por último, se han definido dos límites de seguridad: no recomendar sustancias prohibidas ni tratamientos médicos, y redirigir amablemente al usuario cuando la pregunta queda fuera del dominio. Este último comportamiento se comprobó en los ejemplos documentados, donde ante la pregunta "¿Cuál es la capital de Francia?" el agente respondió correctamente dentro de su rol sin salirse del dominio.

---

## 6. Arquitectura del Grafo (LangGraph)

```
┌─────────────┐
│   USUARIO   │
│  (pregunta) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│           ESTADO DEL AGENTE             │
│  { messages: [...], context: "" }       │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   RETRIEVE  │  <- Nodo 1
│             │     Toma el último mensaje del usuario
│  ChromaDB   │     Busca los 4 chunks más relevantes
│  (k=4)      │     por similitud semántica
└──────┬──────┘
       │  context = chunks recuperados
       ▼
┌─────────────┐
│  GENERATE   │  <- Nodo 2
│             │     Construye: SystemPrompt + contexto RAG
│   Gemini    │     + historial completo de mensajes
│  2.5 Flash  │     Genera la respuesta final
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     END     │
└─────────────┘
       │
  Respuesta añadida al historial (MemorySaver)
  -> disponible en el siguiente turno
```

El grafo está compuesto por dos nodos que se ejecutan de forma secuencial. El primer nodo, `retrieve`, recibe la pregunta del usuario y consulta ChromaDB para recuperar los 4 fragmentos de texto más relevantes mediante similitud semántica. El segundo nodo, `generate`, recibe esos fragmentos como contexto y los combina con el system prompt y el historial completo de mensajes para generar la respuesta con Gemini.

La memoria de conversación se implementa mediante `MemorySaver` de LangGraph, que persiste el estado completo entre invocaciones usando un `thread_id` fijo por sesión. De esta forma, cada nueva pregunta recibe el historial acumulado, permitiendo referencias al contexto anterior sin que el usuario tenga que repetirlo.

---

## 7. Dependencias

```txt
langchain
langchain-google-genai
langchain-chroma
langchain-community
langchain-text-splitters
langgraph
chromadb
pypdf
python-dotenv
google-generativeai
streamlit
```
---

## Estructura del Proyecto

```
proyecto_gym/
├── .env                        <- API Key (no incluir en Git)
├── .gitignore
├── README.md
├── notebook.ipynb              <- Entregable principal
├── app.py                      <- Interfaz web Streamlit
├── data/
│   ├── els50exercicisdeforsamesutilitzats.pdf
│   ├── ENCICLOPEDIA-DE-EJERCICIOS-ACTUALIZADO.pdf
│   └── Libbys-Guia-Nutricion-Deportiva.pdf
└── chroma_db/                  <- Índice vectorial generado automáticamente
```

---

Carlos Checa Moreno
Proyecto Final - IA Generativa  
Módulo: Asistente Experto con Gemini, RAG y Agentes
