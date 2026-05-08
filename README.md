# 🏋️ Agente Experto en Musculación y Fitness

Asistente conversacional especializado en entrenamiento de fuerza, hipertrofia y nutrición deportiva, construido con Google Gemini, LangGraph y ChromaDB mediante técnicas de Retrieval-Augmented Generation (RAG).

---

## 1. Descripción Breve

Este proyecto implementa un agente experto capaz de responder preguntas sobre musculación y fitness basándose en una base de conocimiento vectorial propia. El agente combina recuperación semántica de documentos (RAG) con el modelo de lenguaje Gemini, mantiene memoria de conversación entre turnos y está diseñado para comportarse como un entrenador personal real profesional, cercano y basandose siempre en evidencia.

---

## 2. Dominio Elegido

El agente está especializado en **musculación y fitness**. Los temas que cubre son:

- **Técnica de ejercicios**: sentadilla, peso muerto, press de banca, dominadas, remo y los 50 ejercicios de fuerza más utilizados
- **Entrenamiento**: principios de hipertrofia, frecuencia, volumen, series y repeticiones, estructuras de rutina (Full Body, PPL, Torso-Pierna)
- **Nutrición deportiva**: cálculo de calorías, macronutrientes, timing nutricional, suplementación con evidencia científica
- **Programación**: periodización, sobrecarga progresiva, semanas de descarga
- **Recuperación**: importancia del sueño, señales de sobreentrenamiento, descanso entre sesiones

---

## 3. Stack Tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Google Gemini 2.0 Flash |
| Embeddings | `models/gemini-embedding-001` |
| Base de conocimiento vectorial | ChromaDB (persistida en `./chroma_db`) |
| Framework del agente | LangGraph + LangChain |
| Memoria de conversación | `MemorySaver` de LangGraph |
| Carga de documentos | `PyPDFLoader` de LangChain Community |
| Entorno | Jupyter Notebook / VS Code |

---

## 4. Guía de Ejecución

### Requisitos previos

- Python 3.10+
- API Key de Google Gemini (obtener en https://aistudio.google.com/app/apikey)

### Instalación

```bash
# 1. Clonar o descomprimir el proyecto
cd proyecto_gym

# 2. Instalar dependencias
pip install langchain langchain-google-genai langchain-chroma langchain-community \
            langchain-text-splitters langgraph chromadb pypdf python-dotenv \
            google-generativeai
```

### Configuración de la API Key

Crear un archivo `.env` en la raíz del proyecto:

```
GOOGLE_API_KEY=tu_clave_aqui
```

### Ejecución

1. Colocar los 3 PDFs en la carpeta `data/`:
   - `els50exercicisdeforsamesutilitzats.pdf`
   - `ENCICLOPEDIA-DE-EJERCICIOS-ACTUALIZADO.pdf`
   - `Libbys-Guia-Nutricion-Deportiva.pdf`

2. Abrir `notebook.ipynb` en Jupyter o VS Code

3. Ejecutar todas las celdas en orden (`Kernel → Restart & Run All`)

4. Usar la celda de chat interactivo al final del notebook para conversar con el agente

> **Nota:** La primera ejecución indexa los documentos en ChromaDB (puede tardar 2-3 minutos por los límites de la API gratuita). Las ejecuciones siguientes cargan el índice ya existente directamente.

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

### Decisiones de diseño

**Definición clara del rol** — Se establece explícitamente que el agente es un "entrenador personal y nutricionista deportivo experto". Esto condiciona el tono, el vocabulario y el nivel de profundidad de las respuestas. Sin esta definición, el modelo tendería a respuestas genéricas.

**Jerarquía de fuentes: contexto RAG primero** — La regla más importante del prompt es que el agente debe responder siempre basándose primero en el contexto recuperado. Esto garantiza que el RAG sea el motor real de las respuestas y no un adorno. Cuando el contexto no es suficiente, se indica explícitamente para que el usuario sepa cuándo la respuesta viene de conocimiento general del modelo.

**Tono profesional pero cercano** — Un agente de fitness que hable de forma excesivamente técnica o clínica resulta poco útil en la práctica. El tono de "entrenador real" hace las respuestas más accionables y naturales.

**Mención obligatoria del descanso** — El descanso y la recuperación son tan importantes como el entrenamiento pero frecuentemente olvidados. Incluirlo como regla garantiza que el agente siempre dé una visión completa y no solo la parte "vistosa" del entrenamiento.

**Límites de seguridad** — Prohibir recomendaciones de sustancias prohibidas o tratamientos médicos protege al usuario y delimita el ámbito de responsabilidad del agente.

**Redirección fuera del dominio** — En lugar de responder a cualquier pregunta (lo que degradaría la especialización), el agente redirige amablemente hacia su área de expertise. Esto se comprobó en los ejemplos: ante "¿cuál es la capital de Francia?" el agente respondió correctamente dentro de su rol.

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
│   RETRIEVE  │  ← Nodo 1
│             │    Toma el último mensaje del usuario
│  ChromaDB   │    Busca los 4 chunks más relevantes
│  (k=4)      │    por similitud semántica (cosine)
└──────┬──────┘
       │  context = chunks recuperados
       ▼
┌─────────────┐
│  GENERATE   │  ← Nodo 2
│             │    Construye: SystemPrompt + contexto RAG
│   Gemini    │    + historial completo de mensajes
│  2.0 Flash  │    Genera la respuesta final
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     END     │
└─────────────┘
       │
  Respuesta añadida al historial (MemorySaver)
  → disponible en el siguiente turno
```

**Flujo de memoria:** LangGraph persiste el estado completo (`messages`) entre invocaciones mediante `MemorySaver` con un `thread_id` fijo por sesión. Cada nueva pregunta recibe el historial completo, lo que permite referencias a respuestas anteriores sin que el usuario tenga que repetir el contexto.

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
```

Instalar todo con:

```bash
pip install langchain langchain-google-genai langchain-chroma langchain-community \
            langchain-text-splitters langgraph chromadb pypdf python-dotenv \
            google-generativeai
```

---

## Estructura del Proyecto

```
proyecto_gym/
├── .env                        ← API Key (no incluir en Git)
├── .gitignore
├── README.md
├── notebook.ipynb              ← Entregable principal
├── data/
│   ├── els50exercicisdeforsamesutilitzats.pdf
│   ├── ENCICLOPEDIA-DE-EJERCICIOS-ACTUALIZADO.pdf
│   └── Libbys-Guia-Nutricion-Deportiva.pdf
└── chroma_db/                  ← Índice vectorial generado automáticamente
```

---

## Autor
Carlos Checa Moreno
Proyecto Final — IA Generativa  
Módulo: Asistente Experto con Gemini, RAG y Agentes
