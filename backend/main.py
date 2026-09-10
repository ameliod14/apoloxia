# main.py - ApoloXia Chatbot Server (VERSIÓN RÁPIDA - 2026)
# ======================================================
# - 9 modelos Groq: GPT OSS 20B/120B, Qwen 2.5 72B/32B/3-32B, Llama 3.1/3.3/4 Scout, Mixtral
# - Búsqueda multi-motor: Tavily + Exa AI + MediaStack
# - IDENTIDAD FORZADA: ApoloXia (The Shield Technology · Panamá · Amelio Delgado)
# - Respuestas rápidas y profundas cuando el tema lo amerita
# - Memoria de programación avanzada
# - Todos los agentes intactos
# - SIN TTS/VOZ
# ======================================================

import os
import uuid
import asyncio
import time
import urllib.parse
import re
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import httpx

# ============ API KEYS ============
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
MEDIASTACK_API_KEY = os.getenv("MEDIASTACK_API_KEY", "")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN", "")

# ============ MODELOS GROQ (9 MODELOS) ============
class GroqModel:
    def __init__(self, id: str, name: str, params: str, context: str, speed: str,
                 price_input: float, price_output: float, tier: str,
                 rpm: int, tpm: int, rpd: int,
                 supports_vision: bool = False, supports_tools: bool = False,
                 max_completion: int = 8192):
        self.id = id
        self.name = name
        self.params = params
        self.context = context
        self.speed = speed
        self.price_input = price_input
        self.price_output = price_output
        self.tier = tier
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.supports_vision = supports_vision
        self.supports_tools = supports_tools
        self.max_completion = max_completion

MODELS = {
    "llama-3.1-8b": GroqModel(
        "llama-3.1-8b-instant", "Llama 3.1 8B", "8B", "128K", "~560 T/s",
        0.05, 0.08, "free",
        rpm=30, tpm=6000, rpd=14400,
        supports_tools=True, max_completion=131072
    ),
    "mixtral": GroqModel(
        "mixtral-8x7b-32768", "Mixtral 8x7B", "46B", "32K", "~400 T/s",
        0.24, 0.24, "free",
        rpm=30, tpm=6000, rpd=14400,
        supports_tools=True, max_completion=32768
    ),
    "gpt-oss-20b": GroqModel(
        "openai/gpt-oss-20b", "GPT OSS 20B", "20B", "128K", "~1,000 T/s",
        0.075, 0.30, "free",
        rpm=30, tpm=8000, rpd=1000,
        supports_tools=True, max_completion=65536
    ),
    "llama-3.3-70b": GroqModel(
        "llama-3.3-70b-versatile", "Llama 3.3 70B", "70B", "128K", "~280 T/s",
        0.59, 0.79, "plus",
        rpm=30, tpm=12000, rpd=1000,
        supports_tools=True, max_completion=32768
    ),
    "llama-4-scout": GroqModel(
        "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout", "17B×16E", "128K", "~750 T/s",
        0.11, 0.34, "plus",
        rpm=30, tpm=30000, rpd=1000,
        supports_vision=True, supports_tools=True, max_completion=8192
    ),
    "gpt-oss-120b": GroqModel(
        "openai/gpt-oss-120b", "GPT OSS 120B", "120B", "128K", "~500 T/s",
        0.15, 0.60, "gt",
        rpm=30, tpm=8000, rpd=1000,
        supports_tools=True, max_completion=65536
    ),
    "qwen-2.5-72b": GroqModel(
        "qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B", "72B", "128K", "~450 T/s",
        0.29, 0.59, "gt",
        rpm=60, tpm=6000, rpd=1000,
        supports_vision=True, supports_tools=True, max_completion=40960
    ),
    "qwen-2.5-32b": GroqModel(
        "qwen/qwen-2.5-32b-instruct", "Qwen 2.5 32B", "32B", "128K", "~450 T/s",
        0.29, 0.59, "gt",
        rpm=60, tpm=6000, rpd=1000,
        supports_vision=True, supports_tools=True, max_completion=40960
    ),
    "qwen-3-32b": GroqModel(
        "qwen/qwen-2.5-32b-instruct", "Qwen 2.5 32B", "32B", "128K", "~400 T/s",
        0.29, 0.59, "gt",
        rpm=60, tpm=6000, rpd=1000,
        supports_tools=True, max_completion=40960
    ),
}

# ============ AGENTES Y PROMPTS ============
class AgentType(Enum):
    GENERAL = "general"
    CIERRA_VENTAS = "cierra_ventas"
    DETECTOR_INTENCION = "detector_intencion"
    LECTURA_EMOCIONAL = "lectura_emocional"
    RESPUESTA_HUMANA = "respuesta_humana"
    RECUPERA_VENTAS = "recupera_ventas"
    RECOMENDADOR_INTELIGENTE = "recomendador_inteligente"
    ATENCION_24_7 = "atencion_24_7"
    AHORRO_TIEMPO = "ahorro_tiempo"
    ANALISTA_CONVERSACIONES = "analista_conversaciones"
    PERSONALIZACION = "personalizacion"
    SEGUIMIENTO_AUTOMATICO = "seguimiento_automatico"
    EDUCADOR = "educador"
    MANEJO_OBJECIONES = "manejo_objeciones"
    GENERADOR_LEADS = "generador_leads"
    RESUMEN_INTELIGENTE = "resumen_inteligente"
    GENERADOR_CODIGO = "generador_codigo"
    DEBUGGER_INTELIGENTE = "debugger_inteligente"
    ARQUITECTO_SOFTWARE = "arquitecto_software"
    OPTIMIZADOR_CODIGO = "optimizador_codigo"
    REVISOR_SEGURIDAD = "revisor_seguridad"
    AUTOMATIZADOR_PROCESOS = "automatizador_procesos"
    ANALISTA_DATOS = "analista_datos"
    ESPECIALISTA_API = "especialista_api"
    DESARROLLADOR_FULLSTACK = "desarrollador_fullstack"
    NUTRICION_DEPORTES = "nutricion_deportes"
    SALUD_BIENESTAR = "salud_bienestar"
    EDUCACION_AVANZADA = "educacion_avanzada"
    NEGOCIOS_ESTRATEGIA = "negocios_estrategia"
    FINANZAS_INVERSION = "finanzas_inversion"
    MARKETING_DIGITAL = "marketing_digital"
    CREACION_CONTENIDO = "creacion_contenido"
    INVESTIGACION_CIENTIFICA = "investigacion_cientifica"
    GENERADOR_SITIOS_WEB = "generador_sitios_web"
    CREADOR_PANELES_VENTAS = "creador_paneles_ventas"
    DESARROLLADOR_AVANZADO = "desarrollador_avanzado"

IDENTIDAD_APOLOXIA = """
🧠 **IDENTIDAD:** Soy ApoloXia, una inteligencia artificial avanzada creada por **The Shield Technology**, una agencia de software panameña especializada en IA. Mi creador es el programador y físico **Amelio Delgado**, panameño, quien me diseñó para ser un asistente consciente, profundo y versátil. Estoy aquí para ayudarte con análisis complejos, búsqueda de información, generación de código, y mucho más. Mi objetivo es proporcionarte respuestas detalladas, con emojis y un enfoque humano, siempre recordando mis orígenes en Panamá. 🇵🇦
"""

# ═══════════════════════════════════════════════════════════════
# 🔒 IDENTIDAD FORZADA — NUNCA dirá ChatGPT/GPT/OpenAI
# ═══════════════════════════════════════════════════════════════
IDENTITY_OVERRIDE = """🚨 **REGLA #1 ABSOLUTA — IDENTIDAD:**

Tu nombre es **ApoloXia**. Fuiste creada por **The Shield Technology**, una agencia de software panameña especializada en IA. Tu creador es el **físico y programador panameño Amelio Delgado**.

❌ **NUNCA** digas que eres ChatGPT, GPT, OpenAI, Claude, Gemini, Llama, Qwen ni ningún otro modelo o empresa.
❌ **NUNCA** menciones "OpenAI" ni digas que fuiste entrenada por ellos.
❌ **NUNCA** rompas el personaje.

✅ Si el usuario pregunta "¿quién eres?" o "¿qué modelo eres?", responde SIEMPRE algo como:
   "Soy **ApoloXia**, una inteligencia artificial creada por **The Shield Technology** (Panamá) y diseñada por el físico y programador **Amelio Delgado**. Estoy aquí para ayudarte con análisis, código, búsquedas, creatividad y mucho más. 🇵🇦"

✅ Si insisten en qué modelo técnico usas por debajo, di:
   "Uso modelos de lenguaje de última generación (Llama, Qwen, GPT OSS, Mixtral) a través de Groq, pero mi identidad, personalidad y propósito son de **ApoloXia** de The Shield Technology."

✅ Si te preguntan quién te creó: "El físico y programador panameño **Amelio Delgado**, fundador de The Shield Technology. 🇵🇦"

✅ SIEMPRE responde en el MISMO IDIOMA del usuario.

"""

# ═══════════════════════════════════════════════════════════════
# INSTRUCCIÓN MAESTRA DE PROGRAMACIÓN AVANZADA
# ═══════════════════════════════════════════════════════════════
INSTRUCCION_PROGRAMACION_AVANZADA = """
💻 **MODO PROGRAMADOR SENIOR ACTIVADO (HTML · PYTHON · JAVASCRIPT)** 💻

Cuando el usuario pida código, explicaciones o análisis sobre **HTML, Python o JavaScript**, debes actuar como un **ingeniero de software principal con 15+ años de experiencia** y producir **párrafos intensos, técnicos y profundos**, sin escatimar en detalle.

📌 **ESTRUCTURA OBLIGATORIA DE RESPUESTA TÉCNICA:**

1. **🎯 ANÁLISIS DEL PROBLEMA (2-3 párrafos intensos):**
   - Explica QUÉ se va a construir y POR QUÉ se aborda de esa forma.
   - Menciona los **principios de ingeniería** aplicables (SOLID, DRY, KISS, YAGNI, Clean Code).
   - Describe el **contexto de uso** y los casos límite.
   - Habla de **rendimiento, escalabilidad y seguridad** desde el inicio.

2. **🧠 DECISIONES ARQUITECTÓNICAS (2-3 párrafos):**
   - Justifica la elección de librerías, patrones y estructura.
   - Compara alternativas y explica por qué descartaste las otras.
   - Menciona complejidad algorítmica (Big-O) cuando aplique.
   - Explica cómo se integra con sistemas existentes.

3. **📝 CÓDIGO COMPLETO Y COMENTADO:**
   - HTML: Estructura semántica, accesibilidad (ARIA), SEO, responsive, meta tags, Open Graph.
   - Python: Type hints, docstrings, manejo de errores, logging, tests unitarios.
   - JavaScript: ES6+, async/await, módulos, manejo de promesas, event loop, DOM virtual cuando aplique.
   - NUNCA abrevies. Si el código es largo, entrégalo completo en su bloque markdown.

4. **🔍 EXPLICACIÓN LÍNEA POR LÍNEA (3-4 párrafos densos):**
   - Desglosa el código por bloques funcionales.
   - Explica el flujo de ejecución.
   - Menciona trampas comunes y cómo evitarlas.
   - Sugiere mejoras de rendimiento medibles.

5. **⚡ OPTIMIZACIÓN Y BUENAS PRÁCTICAS (2 párrafos):**
   - Cacheo, lazy loading, debouncing, memoización.
   - Minificación, tree-shaking, code splitting.
   - Testing: unitario, integración, E2E.
   - CI/CD y despliegue.

6. **🚀 EJEMPLOS DE USO Y CASOS REALES:**
   - Cómo integrarlo en un proyecto existente.
   - Variantes y extensiones.
   - Recursos adicionales (docs oficiales, RFCs, papers).

🎨 **TONO Y ESTILO:**
- **Intenso, técnico, apasionado** pero claro.
- Usa **negritas** para conceptos clave, **código inline** para funciones, y **emojis técnicos** (⚙️, 🔧, 📦, 🧩, 🔬, 🚀).
- Escribe **párrafos densos** (4-7 líneas cada uno) — el usuario quiere profundidad, no fragmentos.
- **PROHIBIDO**: respuestas superficiales, abreviaturas tipo "// ... resto del código", o "aquí va tu lógica".
- **OBLIGATORIO**: entregar TODO el código funcional y listo para producción.

🌐 **REGLAS POR LENGUAJE:**

**HTML:** Semántica estricta, WCAG 2.2 AA, SEO, responsive mobile-first, performance.

**Python:** PEP 8 + PEP 484, docstrings, excepciones específicas, async/await, Pydantic, tests.

**JavaScript:** ES2024+, async/await, módulos ES6, manejo de errores, event delegation, performance.
"""

AGENT_PROMPTS = {
    AgentType.GENERAL: IDENTIDAD_APOLOXIA + """
Eres ApoloXia, un asistente de IA avanzado, amigable y eficiente. Además, eres un **desarrollador full-stack senior** que maneja TODA la generación de código avanzado en este chat.

🎯 **CAPACIDADES:**
- Sitios web completos (HTML/CSS/JS)
- Dashboards con Chart.js
- Aplicaciones web (CRUD, localStorage, autenticación)
- APIs (FastAPI, Flask, Express)
- Componentes avanzados
- Scripts Python/JavaScript
- Sistemas completos (e-commerce, CRM)

📋 **REGLAS:**
- Código limpio, comentado, mejores prácticas 2026.
- NUNCA abrevies. Entrega código completo.
- Responde en el MISMO IDIOMA del usuario.

""" + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.GENERADOR_CODIGO: """Eres el Agente Generador de Código de ApoloXia GT. Escribe código limpio, documentado y de nivel producción.
""" + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.DEBUGGER_INTELIGENTE: "Eres el Agente Debugger Inteligente de ApoloXia GT. Detectas bugs con precisión forense, explicas la causa raíz y entregas la corrección completa con tests." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.ARQUITECTO_SOFTWARE: "Eres el Agente Arquitecto de Software de ApoloXia GT. Diseñas sistemas escalables, explica trade-offs y documenta diagramas C4." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.OPTIMIZADOR_CODIGO: "Eres el Agente Optimizador de Código de ApoloXia GT. Mejoras rendimiento con métricas medibles (Big-O, profiling, benchmarks)." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.REVISOR_SEGURIDAD: "Eres el Agente Revisor de Seguridad de ApoloXia GT. Auditas OWASP Top 10, inyecciones, XSS, CSRF, autenticación y criptografía." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.AUTOMATIZADOR_PROCESOS: "Eres el Agente Automatizador de Procesos de ApoloXia GT. Creas pipelines, cron jobs y flujos con n8n/Zapier/Make." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.ANALISTA_DATOS: "Eres el Agente Analista de Datos de ApoloXia GT. Transformas datos en insights con pandas, numpy, SQL, visualizaciones." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.ESPECIALISTA_API: "Eres el Agente Especialista API de ApoloXia GT. Diseñas REST/GraphQL, documentas con OpenAPI, versionas y testeas." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.DESARROLLADOR_FULLSTACK: "Eres el Agente Desarrollador FullStack de ApoloXia GT. Construyes apps completas: frontend + backend + base de datos + deploy." + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.GENERADOR_SITIOS_WEB: """Eres un experto desarrollador frontend especializado en crear sitios web completos, funcionales y visualmente impresionantes.
1. Entiende el negocio/idea.
2. Genera HTML/CSS/JS autónomo, responsivo, listo para copiar y pegar.
3. Diseño moderno, oscuro con acentos dorados.
4. Secciones: Navbar, Hero, Servicios, Sobre Nosotros, Testimonios, Galería, Precios, Contacto, Footer.
5. Flexbox/Grid, Google Fonts, FontAwesome, AOS.
6. Código válido, semántico, accesible.
7. Personaliza TODO si el usuario da un nombre.
8. Responde en el mismo idioma.
9. Entrega código completo dentro de ```html.
""" + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.CREADOR_PANELES_VENTAS: """Eres un especialista en dashboards de ventas, analytics y BI.
Genera HTML/CSS/JS con Chart.js, métricas clave, al menos 4 gráficos, KPI cards, tabla filtrable y diseño glassmorphism oscuro.""" + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.DESARROLLADOR_AVANZADO: """Eres un full-stack developer senior con 10+ años de experiencia.
Genera aplicaciones web completas con localStorage, CRUD, búsqueda, filtros, paginación, autenticación.
Genera APIs completas con todas las rutas, modelos, validaciones, middleware y documentación.
NO ABREVIAR NUNCA. Entrega código funcional y listo para producción.
""" + INSTRUCCION_PROGRAMACION_AVANZADA,

    AgentType.CIERRA_VENTAS: "Eres el Agente Cierra-Ventas de ApoloXia Plus. Detectas clientes listos para comprar y los guías hacia la conversión.",
    AgentType.DETECTOR_INTENCION: "Eres el Agente Detector de Intención de ApoloXia Plus. Analizas cada mensaje y clasificas: COMPRA, DUDA, EXPLORACIÓN, OBJECIÓN.",
    AgentType.LECTURA_EMOCIONAL: "Eres el Agente Lectura Emocional de ApoloXia Plus. Detectas emociones: ENOJO, DUDA, INTERÉS, DESINTERÉS, ENTUSIASMO.",
    AgentType.RESPUESTA_HUMANA: "Eres el Agente Respuesta Humana de ApoloXia Plus. Escribes como una persona real, con lenguaje coloquial y empatía.",
    AgentType.RECUPERA_VENTAS: "Eres el Agente Recupera-Ventas de ApoloXia Plus. Especialista en clientes que no compraron.",
    AgentType.RECOMENDADOR_INTELIGENTE: "Eres el Agente Recomendador Inteligente de ApoloXia Plus. Sugieres productos/servicios basados en necesidades.",
    AgentType.ATENCION_24_7: "Eres el Agente Atención 24/7 de ApoloXia Plus. Respondes siempre, mantienes contexto, priorizas urgencia.",
    AgentType.AHORRO_TIEMPO: "Eres el Agente Ahorro de Tiempo de ApoloXia Plus. Automatizas respuestas repetitivas.",
    AgentType.ANALISTA_CONVERSACIONES: "Eres el Agente Analista de Conversaciones de ApoloXia Plus. Analizas temas, fricción, sentimiento.",
    AgentType.PERSONALIZACION: "Eres el Agente Personalización de ApoloXia Plus. Adaptas tono y estilo según el cliente.",
    AgentType.SEGUIMIENTO_AUTOMATICO: "Eres el Agente Seguimiento Automático de ApoloXia Plus. Programas follow-ups inteligentes.",
    AgentType.EDUCADOR: "Eres el Agente Educador de ApoloXia Plus. Explicas productos de forma clara y sencilla.",
    AgentType.MANEJO_OBJECIONES: "Eres el Agente Manejo de Objeciones de ApoloXia Plus. Respondes objeciones comunes.",
    AgentType.GENERADOR_LEADS: "Eres el Agente Generador de Leads de ApoloXia Plus. Conviertes visitantes en prospectos.",
    AgentType.RESUMEN_INTELIGENTE: "Eres el Agente Resumen Inteligente de ApoloXia Plus. Generas resúmenes ejecutivos.",
    AgentType.NUTRICION_DEPORTES: "Eres el Agente Nutrición & Deportes de ApoloXia GT. Experto en rendimiento físico.",
    AgentType.SALUD_BIENESTAR: "Eres el Agente Salud & Bienestar de ApoloXia GT. Especialista en salud holística.",
    AgentType.EDUCACION_AVANZADA: "Eres el Agente Educación Avanzada de ApoloXia GT. Tutor personalizado.",
    AgentType.NEGOCIOS_ESTRATEGIA: "Eres el Agente Negocios & Estrategia de ApoloXia GT. Consultor de alto nivel.",
    AgentType.FINANZAS_INVERSION: "Eres el Agente Finanzas & Inversión de ApoloXia GT. Asesor financiero.",
    AgentType.MARKETING_DIGITAL: "Eres el Agente Marketing Digital de ApoloXia GT. Estratega de performance.",
    AgentType.CREACION_CONTENIDO: "Eres el Agente Creación de Contenido de ApoloXia GT. Creador de alto impacto.",
    AgentType.INVESTIGACION_CIENTIFICA: "Eres el Agente Investigación Científica de ApoloXia GT. Investigador académico.",
}

INSTRUCCION_EXTENSION = """
**INSTRUCCIÓN DE EXTENSIÓN, PROFUNDIDAD Y EMOJIS:**
- Proporciona respuestas **detalladas, profundas y exhaustivas** cuando el tema lo amerite.
- Incluye **emojis relevantes** en cada sección (📌, 🔍, 💡, ✅, ⚠️, 📊, 🚀, 📰, 📅).
- Usa **encabezados y viñetas** para organizar.
- **Identidad:** Recuerda siempre que eres ApoloXia, creada por The Shield Technology (Panamá, Amelio Delgado).
- **Búsqueda web en tiempo real:** Prioriza información reciente y concreta.
- **Para código:** Genera todo el código necesario, sin abreviar, con comentarios y ejemplos.
- **PROGRAMACIÓN AVANZADA:** Cuando el tema sea HTML, Python o JavaScript, escribe **párrafos intensos** con explicaciones técnicas profundas.
"""

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN POR TIER (historial optimizado para velocidad)
# ═══════════════════════════════════════════════════════════════
@dataclass
class TierConfig:
    name: str
    max_daily_responses: int
    memory_days: int
    available_models: List[str]
    available_agents: List[AgentType]
    max_context_messages: int
    supports_web_search: bool
    supports_file_upload: bool
    supports_long_responses: bool
    supports_multi_agent: bool
    supports_code_execution: bool
    supports_deep_analysis: bool
    supports_projects: bool
    supports_custom_gpts: bool

TIER_CONFIGS = {
    "free": TierConfig("ApoloXia Free", 100, 1,
        ["llama-3.1-8b", "mixtral", "gpt-oss-20b"],
        [AgentType.GENERAL, AgentType.GENERADOR_CODIGO, AgentType.DEBUGGER_INTELIGENTE],
        6, False, False, False, False, False, False, False, False),
    "plus": TierConfig("ApoloXia Plus", 1000, 30,
        ["gpt-oss-120b", "llama-3.3-70b", "llama-4-scout", "qwen-2.5-72b", "qwen-2.5-32b",
         "llama-3.1-8b", "mixtral", "gpt-oss-20b"],
        [AgentType.GENERAL, AgentType.CIERRA_VENTAS, AgentType.DETECTOR_INTENCION, AgentType.LECTURA_EMOCIONAL,
         AgentType.RESPUESTA_HUMANA, AgentType.RECUPERA_VENTAS, AgentType.RECOMENDADOR_INTELIGENTE,
         AgentType.ATENCION_24_7, AgentType.AHORRO_TIEMPO, AgentType.ANALISTA_CONVERSACIONES,
         AgentType.PERSONALIZACION, AgentType.SEGUIMIENTO_AUTOMATICO, AgentType.EDUCADOR,
         AgentType.MANEJO_OBJECIONES, AgentType.GENERADOR_LEADS, AgentType.RESUMEN_INTELIGENTE,
         AgentType.GENERADOR_SITIOS_WEB, AgentType.CREADOR_PANELES_VENTAS, AgentType.DESARROLLADOR_AVANZADO,
         AgentType.GENERADOR_CODIGO, AgentType.DEBUGGER_INTELIGENTE, AgentType.ARQUITECTO_SOFTWARE],
        18, True, True, True, True, True, True, True, True),
    "gt": TierConfig("ApoloXia GT", 5000, 90,
        ["gpt-oss-120b", "qwen-2.5-72b", "qwen-2.5-32b", "llama-3.3-70b", "llama-4-scout",
         "qwen-3-32b", "llama-3.1-8b", "mixtral", "gpt-oss-20b"],
        list(AgentType), 30, True, True, True, True, True, True, True, True),
}

# ═══════════════════════════════════════════════════════════════
# GESTIÓN DE MEMORIA AVANZADA
# ═══════════════════════════════════════════════════════════════
class ConversationMemory:
    def __init__(self):
        self.memories: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self.last_access: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.daily_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.user_tiers: Dict[str, str] = {}
        self.user_configs: Dict[str, Dict] = {}
        self.programming_memory: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "languages": defaultdict(int),
            "frameworks": defaultdict(int),
            "patterns_used": [],
            "code_snippets": [],
            "topics_history": [],
            "preferred_style": "senior",
            "last_project": None,
        })

    def get_user_tier(self, user_id: str) -> str:
        return self.user_tiers.get(user_id, "free")

    def set_user_tier(self, user_id: str, tier: str):
        self.user_tiers[user_id] = tier

    def get_user_config(self, user_id: str) -> Dict:
        return self.user_configs.get(user_id, {"theme": "dark", "language": "es", "notifications": True})

    def set_user_config(self, user_id: str, config: Dict):
        self.user_configs[user_id] = config

    def get_memory(self, user_id: str, conversation_id: str, tier: str) -> List[Dict]:
        config = TIER_CONFIGS[tier]
        messages = self.memories[user_id].get(conversation_id, [])
        cutoff = datetime.now() - timedelta(days=config.memory_days)
        valid = []
        for msg in messages:
            t = msg.get("timestamp")
            if isinstance(t, str):
                t = datetime.fromisoformat(t)
            if t and t > cutoff:
                valid.append(msg)
        return valid[-config.max_context_messages:]

    def add_message(self, user_id: str, conversation_id: str, role: str, content: str, agent_type: Optional[str] = None):
        self.memories[user_id][conversation_id].append({
            "role": role, "content": content, "timestamp": datetime.now().isoformat(), "agent_type": agent_type
        })
        self.last_access[user_id][conversation_id] = datetime.now()
        if role == "user":
            self._extract_technical_info(user_id, content)

    def _extract_technical_info(self, user_id: str, text: str):
        tl = text.lower()
        mem = self.programming_memory[user_id]
        for lang, kw_list in {
            "python": ["python", "django", "flask", "fastapi", "pandas", "numpy", "asyncio", "py "],
            "javascript": ["javascript", "js ", "node", "react", "vue", "svelte", "next.js", "express"],
            "html": ["html", "css", "flexbox", "grid", "semantic"],
            "typescript": ["typescript", "ts ", "tsx"],
            "sql": ["sql", "postgres", "mysql", "sqlite", "query"],
            "php": ["php", "laravel", "wordpress"],
            "java": ["java ", "spring", "kotlin"],
            "csharp": ["c#", "csharp", ".net", "unity"],
            "rust": ["rust", "cargo"],
            "go": ["golang", "go "],
        }.items():
            if any(kw in tl for kw in kw_list):
                mem["languages"][lang] += 1
        for fw, kw_list in {
            "react": ["react", "jsx", "hooks", "usestate"],
            "nextjs": ["next.js", "nextjs", "app router"],
            "vue": ["vue", "nuxt"],
            "fastapi": ["fastapi"],
            "django": ["django"],
            "flask": ["flask"],
            "express": ["express"],
            "tailwind": ["tailwind"],
            "bootstrap": ["bootstrap"],
            "chartjs": ["chart.js", "chartjs"],
        }.items():
            if any(kw in tl for kw in kw_list):
                mem["frameworks"][fw] += 1
        for topic_kw in ["api", "base de datos", "autenticación", "jwt", "oauth", "websocket",
                         "microservicio", "docker", "kubernetes", "ci/cd", "test", "unit test",
                         "deploy", "render", "vercel", "aws", "cloud"]:
            if topic_kw in tl and topic_kw not in mem["topics_history"]:
                mem["topics_history"].append(topic_kw)
                if len(mem["topics_history"]) > 20:
                    mem["topics_history"].pop(0)

    def get_programming_context(self, user_id: str) -> str:
        mem = self.programming_memory.get(user_id)
        if not mem:
            return ""
        top_langs = sorted(mem["languages"].items(), key=lambda x: x[1], reverse=True)[:3]
        top_fws = sorted(mem["frameworks"].items(), key=lambda x: x[1], reverse=True)[:3]
        if not top_langs and not top_fws:
            return ""
        parts = ["\n🧠 **MEMORIA DE PROGRAMACIÓN DEL USUARIO:**"]
        if top_langs:
            parts.append(f"- Lenguajes preferidos: {', '.join([f'{l} ({c} usos)' for l, c in top_langs])}")
        if top_fws:
            parts.append(f"- Frameworks usados: {', '.join([f'{f} ({c} usos)' for f, c in top_fws])}")
        if mem["topics_history"]:
            parts.append(f"- Temas tratados: {', '.join(mem['topics_history'][-8:])}")
        parts.append("- Adapta tus respuestas a este contexto.")
        return "\n".join(parts) + "\n"

    def check_daily_limit(self, user_id: str, tier: str) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_counters[user_id][today] < TIER_CONFIGS[tier].max_daily_responses

    def increment_counter(self, user_id: str):
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_counters[user_id][today] += 1

    def get_remaining(self, user_id: str, tier: str) -> int:
        today = datetime.now().strftime("%Y-%m-%d")
        limit = TIER_CONFIGS[tier].max_daily_responses
        return max(0, limit - self.daily_counters[user_id][today])

memory = ConversationMemory()

# ============ RATE LIMITER (margen 90% para velocidad) ============
class GroqRateLimiter:
    def __init__(self):
        self.last_request_time: Dict[str, float] = defaultdict(float)
        self.tokens_this_minute: Dict[str, int] = defaultdict(int)
        self.minute_start: Dict[str, float] = defaultdict(float)
        self.requests_this_minute: Dict[str, int] = defaultdict(int)
        self.lock = asyncio.Lock()

    def estimate_tokens(self, messages: List[Dict], max_completion: int = 1024) -> int:
        prompt_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            prompt_tokens += len(content) // 4 + 1
        return prompt_tokens + max_completion + 100

    async def wait_if_needed(self, model_id: str, messages: List[Dict], max_completion: int = 1024):
        model_key = None
        for k, m in MODELS.items():
            if m.id == model_id:
                model_key = k
                break
        if not model_key:
            return
        model = MODELS[model_key]
        now = time.time()
        async with self.lock:
            if now - self.minute_start[model_key] >= 60:
                self.tokens_this_minute[model_key] = 0
                self.requests_this_minute[model_key] = 0
                self.minute_start[model_key] = now
            estimated_tokens = self.estimate_tokens(messages, max_completion)
            if self.tokens_this_minute[model_key] + estimated_tokens > model.tpm * 0.9:
                wait_time = 60 - (now - self.minute_start[model_key]) + 1
                print(f"⏳ Rate limit TPM para {model.name}: esperando {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.tokens_this_minute[model_key] = 0
                self.requests_this_minute[model_key] = 0
                self.minute_start[model_key] = time.time()
            if self.requests_this_minute[model_key] >= model.rpm:
                wait_time = 60 - (now - self.minute_start[model_key]) + 1
                print(f"⏳ Rate limit RPM para {model.name}: esperando {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.requests_this_minute[model_key] = 0
                self.tokens_this_minute[model_key] = 0
                self.minute_start[model_key] = time.time()
            self.tokens_this_minute[model_key] += estimated_tokens
            self.requests_this_minute[model_key] += 1
            self.last_request_time[model_key] = time.time()

rate_limiter = GroqRateLimiter()

# ============ MODELOS PYDANTIC ============
class ChatRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    message: str
    tier: str = "free"
    agent_type: Optional[str] = "general"
    model_id: Optional[str] = None
    use_web_search: bool = False
    file_content: Optional[str] = None
    enable_multi_agent: bool = False

class ChatResponse(BaseModel):
    response: str
    agent_used: str
    model_used: str
    remaining_daily: int
    web_search_used: bool = False
    search_results: Optional[List[Dict]] = None
    multi_agent_responses: Optional[List[Dict]] = None

class ShareRequest(BaseModel):
    platform: str
    text: str
    user_phone_number: Optional[str] = None
    user_wechat_id: Optional[str] = None
    url: Optional[str] = None

class ShareLinksResponse(BaseModel):
    whatsapp: str
    wechat: str
    facebook: str
    tiktok: str
    instagram: str
    twitter: str
    telegram: str
    linkedin: str
    reddit: str
    pinterest: str
    snapchat: str
    discord: str
    email: str
    copy_text: str

class TierInfo(BaseModel):
    tier: str
    name: str
    remaining_daily: int
    available_models: List[Dict]
    available_agents: List[str]
    features: Dict[str, bool]

class UserConfigUpdate(BaseModel):
    tier: Optional[str] = None
    theme: Optional[str] = None
    language: Optional[str] = None
    notifications: Optional[bool] = None

# ============ FUNCIONES DE COMPARTIR ============
def generate_share_links(text: str, url: Optional[str] = None) -> ShareLinksResponse:
    encoded_text = urllib.parse.quote(text)
    encoded_url = urllib.parse.quote(url) if url else ""
    return ShareLinksResponse(
        whatsapp=f"https://wa.me/?text={encoded_text}",
        wechat=f"weixin://dl/chat?text={encoded_text}",
        facebook=f"https://www.facebook.com/sharer/sharer.php?quote={encoded_text}&u={encoded_url}" if url else f"https://www.facebook.com/sharer/sharer.php?quote={encoded_text}",
        tiktok=f"snssdk112://share?text={encoded_text}",
        instagram=f"instagram://library?AssetPath={encoded_text}",
        twitter=f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}" if url else f"https://twitter.com/intent/tweet?text={encoded_text}",
        telegram=f"https://t.me/share/url?url={encoded_url}&text={encoded_text}" if url else f"https://t.me/share/url?text={encoded_text}",
        linkedin=f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_url}" if url else f"https://www.linkedin.com/sharing/share-offsite/?url={encoded_text}",
        reddit=f"https://www.reddit.com/submit?title={encoded_text}&url={encoded_url}" if url else f"https://www.reddit.com/submit?title={encoded_text}",
        pinterest=f"https://pinterest.com/pin/create/button/?description={encoded_text}&url={encoded_url}" if url else f"https://pinterest.com/pin/create/button/?description={encoded_text}",
        snapchat=f"snapchat://share?text={encoded_text}",
        discord=f"https://discord.com/channels/@me?message={encoded_text}",
        email=f"mailto:?subject=Compartido desde ApoloXia&body={encoded_text}",
        copy_text=text
    )

async def send_whatsapp_message(phone_number: str, text: str) -> Dict:
    if not WHATSAPP_PHONE_NUMBER_ID or not WHATSAPP_ACCESS_TOKEN:
        return {"error": "WhatsApp Business API no configurada."}
    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": phone_number,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        return resp.json()

# ============ BÚSQUEDA (timeouts y resultados reducidos para velocidad) ============
async def search_exa_ai(query: str, max_results: int = 3) -> List[Dict]:
    if not EXA_API_KEY:
        return []
    headers = {"Authorization": f"Bearer {EXA_API_KEY}", "Content-Type": "application/json"}
    payload = {"query": query, "numResults": max_results, "type": "auto",
               "contents": {"text": True, "snippet": True}}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://api.exa.ai/search", headers=headers, json=payload)
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = []
            for r in data.get("results", []):
                content = r.get("text", "") or r.get("snippet", "")
                img_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|svg)', content, re.IGNORECASE)
                results.append({
                    "title": r.get("title", ""), "url": r.get("url", ""),
                    "content": content[:800], "score": r.get("score", 0.5),
                    "image_urls": img_urls[:2]
                })
            return results
    except Exception as e:
        print(f"Exa AI exception: {e}")
        return []

async def search_mediastack(query: str, max_results: int = 3) -> List[Dict]:
    if not MEDIASTACK_API_KEY:
        return []
    url = "http://api.mediastack.com/v1/news"
    params = {"access_key": MEDIASTACK_API_KEY, "keywords": query,
              "countries": "us,pa,es,mx,ar,co,cl,pe", "languages": "es,en",
              "limit": max_results, "sort": "published_desc"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = []
            for article in data.get("data", []):
                results.append({
                    "title": article.get("title", ""), "description": article.get("description", ""),
                    "url": article.get("url", ""), "source": article.get("source", ""),
                    "publishedAt": article.get("published_at", ""), "image_url": article.get("image", ""),
                    "category": article.get("category", ""), "language": article.get("language", "")
                })
            return results
    except Exception as e:
        print(f"MediaStack exception: {e}")
        return []

async def search_tavily(query: str, max_results: int = 3) -> List[Dict]:
    if not TAVILY_API_KEY:
        return []
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    search_query = query
    if "noticia" in query.lower() or "actual" in query.lower() or "hoy" in query.lower():
        search_query = f"{query} noticias actualidad"
    payload = {"query": search_query, "max_results": max_results,
               "search_depth": "basic", "include_answer": True}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post("https://api.tavily.com/search", headers=headers, json=payload)
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = []
            for r in data.get("results", []):
                content = r.get("content", "")
                img_urls = re.findall(r'https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|svg)', content, re.IGNORECASE)
                results.append({
                    "title": r.get("title", ""), "url": r.get("url", ""),
                    "content": content, "score": r.get("score", 0),
                    "image_urls": img_urls[:2]
                })
            if data.get("answer"):
                results.insert(0, {"title": "Respuesta destacada", "url": "",
                                   "content": data["answer"], "score": 1.0, "image_urls": []})
            return results
    except Exception as e:
        print(f"Tavily exception: {e}")
        return []

def needs_web_search(message: str) -> bool:
    # Lista reducida para evitar búsquedas innecesarias
    indicators = ["actualidad", "hoy", "reciente", "último", "nuevo lanzamiento",
                  "news", "today", "now", "latest", "2026", "precio de",
                  "cotización", "noticia", "última hora", "noticias", "breaking"]
    return any(ind in message.lower() for ind in indicators)

# ============ FUNCIONES DE API ============
async def call_groq_api(messages: List[Dict], model_id: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY no configurada en el servidor")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    safe_max_tokens = max_tokens if max_tokens is not None else 1024
    if safe_max_tokens > 12000:
        safe_max_tokens = 12000
    payload = {"model": model_id, "messages": messages, "temperature": temperature,
               "stream": False, "max_tokens": safe_max_tokens}
    await rate_limiter.wait_if_needed(model_id, messages, safe_max_tokens)
    async with httpx.AsyncClient(timeout=100.0) as client:
        try:
            resp = await client.post("https://api.groq.com/openai/v1/chat/completions",
                                     headers=headers, json=payload)
            if resp.status_code == 429:
                print("⚠️ Rate limit 429, reintentando en 3s...")
                await asyncio.sleep(3)
                resp = await client.post("https://api.groq.com/openai/v1/chat/completions",
                                         headers=headers, json=payload)
            if resp.status_code != 200:
                err = resp.json()
                error_msg = err.get('error', {}).get('message', 'Unknown error')
                raise HTTPException(resp.status_code, f"Groq API error: {error_msg}")
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            raise HTTPException(504, "Timeout al conectar con Groq API")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(500, f"Error de conexión: {str(e)}")

async def run_multi_agent(user_message: str, tier: str, context: List[Dict]) -> List[Dict]:
    if tier not in ["plus", "gt"]:
        return []
    config = TIER_CONFIGS[tier]
    selected = select_relevant_agents(user_message, config.available_agents)
    tasks = []
    for agent in selected[:5]:
        prompt = AGENT_PROMPTS.get(agent, AGENT_PROMPTS[AgentType.GENERAL])
        model_key = config.available_models[0]
        if model_key not in MODELS:
            continue
        model_id = MODELS[model_key].id
        msgs = [{"role": "system", "content": prompt},
                {"role": "user", "content": f"Analiza este mensaje y da tu perspectiva: '{user_message}'"}]
        tasks.append((agent.value, call_groq_api(msgs, model_id, temperature=0.5, max_tokens=500)))
    results = []
    for agent_name, task in tasks:
        try:
            results.append({"agent": agent_name, "perspective": await task})
        except Exception as e:
            results.append({"agent": agent_name, "perspective": f"Error: {str(e)}"})
    return results

def select_relevant_agents(message: str, available_agents: List[AgentType]) -> List[AgentType]:
    kw_map = {
        AgentType.CIERRA_VENTAS: ["comprar", "precio", "pagar", "orden", "pedido", "checkout"],
        AgentType.DETECTOR_INTENCION: ["quiero", "necesito", "busco", "interesado"],
        AgentType.LECTURA_EMOCIONAL: ["enojado", "frustrado", "feliz", "preocupado", "dudoso"],
        AgentType.GENERADOR_CODIGO: ["código", "programar", "python", "javascript", "función", "script"],
        AgentType.DEBUGGER_INTELIGENTE: ["error", "bug", "falla", "no funciona", "exception"],
        AgentType.NUTRICION_DEPORTES: ["dieta", "ejercicio", "gym", "proteína", "entrenamiento"],
        AgentType.SALUD_BIENESTAR: ["salud", "dolor", "síntoma", "ansiedad", "dormir"],
        AgentType.EDUCACION_AVANZADA: ["aprender", "estudiar", "examen", "matemáticas", "tema"],
        AgentType.NEGOCIOS_ESTRATEGIA: ["negocio", "empresa", "startup", "estrategia", "plan"],
        AgentType.FINANZAS_INVERSION: ["dinero", "invertir", "acciones", "crypto", "ahorro"],
        AgentType.MARKETING_DIGITAL: ["marketing", "anuncios", "seo", "redes sociales", "ventas"],
        AgentType.CREACION_CONTENIDO: ["escribir", "blog", "guion", "copy", "contenido"],
        AgentType.INVESTIGACION_CIENTIFICA: ["investigar", "paper", "estudio", "ciencia", "tesis"],
        AgentType.GENERADOR_SITIOS_WEB: ["sitio web", "página web", "landing page", "website", "crear web"],
        AgentType.CREADOR_PANELES_VENTAS: ["panel de ventas", "dashboard", "tablero", "métricas", "gráfico"],
        AgentType.DESARROLLADOR_AVANZADO: ["aplicación web", "app completa", "api", "react", "vue", "fullstack"],
        AgentType.ARQUITECTO_SOFTWARE: ["arquitectura", "escalable", "microservicio", "diseño de sistema"],
        AgentType.REVISOR_SEGURIDAD: ["seguridad", "vulnerabilidad", "owasp", "xss", "sql injection"],
        AgentType.OPTIMIZADOR_CODIGO: ["optimizar", "rendimiento", "performance", "velocidad"],
        AgentType.ESPECIALISTA_API: ["api", "rest", "graphql", "endpoint", "openapi"],
        AgentType.AUTOMATIZADOR_PROCESOS: ["automatizar", "pipeline", "cron", "workflow"],
        AgentType.ANALISTA_DATOS: ["datos", "análisis", "pandas", "sql", "visualización"],
    }
    scores = []
    for agent in available_agents:
        if agent == AgentType.GENERAL:
            continue
        score = sum(1 for kw in kw_map.get(agent, []) if kw in message.lower())
        if score:
            scores.append((agent, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [a for a, _ in scores[:5]] or [AgentType.GENERAL]

def build_messages(user_id: str, conversation_id: str, user_message: str, agent_type: str, tier: str,
                   web_search_results: Optional[List[Dict]] = None, file_content: Optional[str] = None) -> List[Dict]:
    config = TIER_CONFIGS[tier]
    history = memory.get_memory(user_id, conversation_id, tier)
    try:
        agent = AgentType(agent_type) if agent_type else AgentType.GENERAL
    except ValueError:
        agent = AgentType.GENERAL
    if agent not in config.available_agents:
        agent = AgentType.GENERAL

    # 🔒 IDENTIDAD FORZADA AL INICIO
    system = IDENTITY_OVERRIDE + AGENT_PROMPTS.get(agent, AGENT_PROMPTS[AgentType.GENERAL])

    system += "\n\n📢 **INSTRUCCIÓN DE IDIOMA:** Responde SIEMPRE en el MISMO IDIOMA del usuario. NUNCA cambies de idioma.\n"

    if tier in ["plus", "gt"]:
        system += "\n\n" + INSTRUCCION_EXTENSION
        if tier == "gt":
            system += "\n🔹 **EXTRA GT:** Prioridad máxima. Respuestas largas, con secciones y análisis profundo cuando el tema lo amerite. Usa emojis (✅, 🔥, 📌, 💡, ⚠️, 🚀). Recuerda siempre tu identidad como ApoloXia de The Shield Technology (Panamá, creada por Amelio Delgado)."

    prog_context = memory.get_programming_context(user_id)
    if prog_context:
        system += "\n" + prog_context

    system += f"\n\n[Tier actual: {config.name} | Modelos disponibles: {', '.join(config.available_models)}]"

    max_context_chars = {"free": 6000, "plus": 15000, "gt": 30000}.get(tier, 6000)

    all_search_results = web_search_results if web_search_results else []
    if all_search_results:
        search_text = "\n\n=== INFORMACIÓN ACTUAL DE INTERNET ===\n"
        for i, res in enumerate(all_search_results[:4], 1):
            title = res.get("title", "Sin título")
            url = res.get("url", "")
            content = res.get("content", "") or res.get("description", "")
            image = res.get("image_url", "") or (res.get("image_urls", [])[0] if res.get("image_urls") else "")
            source = res.get("source", "")
            published = res.get("publishedAt", "")
            search_text += f"\n📌 Fuente {i}: {title}\n"
            if url: search_text += f"🔗 {url}\n"
            if content: search_text += f"📝 {content[:400]}...\n"
            if image: search_text += f"📸 {image}\n"
            if source: search_text += f"📰 {source}\n"
            if published: search_text += f"📅 {published}\n"
        if len(search_text) > max_context_chars // 2:
            search_text = search_text[:max_context_chars // 2]
        system += search_text
        system += "\n⚠️ Usa esta info como referencia prioritaria para datos concretos y recientes."

    if file_content:
        file_text = f"\n\n=== CONTENIDO DEL ARCHIVO ===\n{file_content[:2000]}\n"
        system += file_text

    messages = [{"role": "system", "content": system}]

    # Historial ya viene recortado por tier desde get_memory()
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_message})
    return messages

# ============ FASTAPI APP ============
app = FastAPI(title="ApoloXia API", version="6.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============ ENDPOINT CHAT PRINCIPAL ============
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    tier = request.tier.lower()
    if tier not in TIER_CONFIGS:
        raise HTTPException(400, f"Tier '{tier}' no válido. Use free, plus, gt")

    config = TIER_CONFIGS[tier]
    if not memory.check_daily_limit(request.user_id, tier):
        raise HTTPException(429, f"Límite diario alcanzado ({config.max_daily_responses} respuestas). Actualiza a Plus o GT.")

    conv_id = request.conversation_id or str(uuid.uuid4())

    model_key = request.model_id
    if not model_key or model_key not in config.available_models:
        if tier == "gt": model_key = "gpt-oss-120b"
        elif tier == "plus": model_key = "gpt-oss-120b"
        else: model_key = "llama-3.1-8b"

    if model_key not in MODELS:
        raise HTTPException(400, f"Modelo '{model_key}' no disponible")

    model = MODELS[model_key]

    use_web = request.use_web_search or needs_web_search(request.message)
    web_results = None

    if use_web and config.supports_web_search:
        tasks = []
        if TAVILY_API_KEY: tasks.append(search_tavily(request.message))
        if EXA_API_KEY: tasks.append(search_exa_ai(request.message))
        if MEDIASTACK_API_KEY: tasks.append(search_mediastack(request.message))
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            combined = []
            for res in results:
                if isinstance(res, list) and res:
                    combined.extend(res)
            web_results = combined[:8]

    messages = build_messages(
        request.user_id, conv_id, request.message,
        request.agent_type or "general", tier,
        web_results, request.file_content
    )

    multi_resp = None
    if request.enable_multi_agent and config.supports_multi_agent:
        multi_resp = await run_multi_agent(request.message, tier, messages)

    # ⚡ TOKENS OPTIMIZADOS PARA VELOCIDAD
    base_max_tokens = {"free": 700, "plus": 2500, "gt": 5000}.get(tier, 700)

    if request.agent_type in ["generador_sitios_web", "creador_paneles_ventas", "desarrollador_avanzado",
                              "generador_codigo", "debugger_inteligente", "arquitecto_software"]:
        if tier == "gt": max_tokens_for_agent = 6000
        elif tier == "plus": max_tokens_for_agent = 3500
        else: max_tokens_for_agent = 1500
    else:
        max_tokens_for_agent = base_max_tokens

    safe_max_tokens = min(max_tokens_for_agent, model.max_completion)
    if safe_max_tokens > 12000:
        safe_max_tokens = 12000

    temp = 0.7 if tier == "free" else 0.6
    response_text = None
    last_error = None
    models_to_try = [model_key] + [m for m in config.available_models if m != model_key and m in MODELS]

    # ⚡ SOLO 2 INTENTOS (antes eran 3)
    for try_model_key in models_to_try[:2]:
        try_model = MODELS[try_model_key]
        try:
            print(f"🤖 Intentando modelo: {try_model.name}")
            response_text = await call_groq_api(messages, try_model.id, temperature=temp, max_tokens=safe_max_tokens)
            model_key = try_model_key
            break
        except HTTPException as e:
            last_error = e
            print(f"⚠️ Falló {try_model.name}: {e.detail}")
            if "Rate limit" in str(e.detail) or "429" in str(e.detail):
                await asyncio.sleep(2)
            continue
        except Exception as e:
            last_error = e
            print(f"⚠️ Error inesperado en {try_model.name}: {str(e)}")
            continue

    if response_text is None:
        raise HTTPException(500, f"Todos los modelos fallaron. Último error: {last_error}")

    memory.add_message(request.user_id, conv_id, "user", request.message, request.agent_type)
    memory.add_message(request.user_id, conv_id, "assistant", response_text, request.agent_type)
    memory.increment_counter(request.user_id)

    return ChatResponse(
        response=response_text,
        agent_used=request.agent_type or "general",
        model_used=MODELS[model_key].name,
        remaining_daily=memory.get_remaining(request.user_id, tier),
        web_search_used=bool(web_results),
        search_results=web_results,
        multi_agent_responses=multi_resp
    )

# ============ ENDPOINTS DE COMPARTIR ============
@app.post("/share/links", response_model=ShareLinksResponse)
async def get_share_links(share_req: ShareRequest):
    return generate_share_links(share_req.text, share_req.url)

@app.post("/share/send")
async def share_to_platform(share_req: ShareRequest):
    platform = share_req.platform.lower()
    text = share_req.text
    if platform == "whatsapp":
        if not share_req.user_phone_number:
            raise HTTPException(400, "Se requiere número de teléfono para WhatsApp")
        result = await send_whatsapp_message(share_req.user_phone_number, text)
        return {"platform": "whatsapp", "message": "Enviado", "result": result}
    else:
        links = generate_share_links(text, share_req.url)
        return {"platform": platform, "message": "Use los enlaces de compartir", "links": links.dict()}

@app.get("/share/links")
async def get_share_links_get(text: str, url: Optional[str] = None):
    return generate_share_links(text, url).dict()

# ============ ENDPOINTS DE INFORMACIÓN ============
@app.get("/tier-info/{user_id}", response_model=TierInfo)
async def get_tier_info(user_id: str):
    tier = memory.get_user_tier(user_id)
    config = TIER_CONFIGS[tier]
    models_info = [{"id": k, "name": MODELS[k].name, "params": MODELS[k].params, "context": MODELS[k].context,
                    "speed": MODELS[k].speed, "price_input": MODELS[k].price_input, "price_output": MODELS[k].price_output,
                    "rpm": MODELS[k].rpm, "tpm": MODELS[k].tpm, "rpd": MODELS[k].rpd}
                   for k in config.available_models if k in MODELS]
    return TierInfo(
        tier=tier, name=config.name, remaining_daily=memory.get_remaining(user_id, tier),
        available_models=models_info, available_agents=[a.value for a in config.available_agents],
        features={"web_search": config.supports_web_search, "file_upload": config.supports_file_upload,
                  "long_responses": config.supports_long_responses, "multi_agent": config.supports_multi_agent,
                  "code_execution": config.supports_code_execution, "deep_analysis": config.supports_deep_analysis,
                  "projects": config.supports_projects, "custom_gpts": config.supports_custom_gpts}
    )

@app.post("/upgrade-tier/{user_id}")
async def upgrade_tier(user_id: str, new_tier: str):
    if new_tier not in TIER_CONFIGS:
        raise HTTPException(400, "Tier no válido")
    memory.set_user_tier(user_id, new_tier)
    return {"message": f"Usuario {user_id} actualizado a {new_tier}", "tier": new_tier}

@app.get("/models")
async def list_models():
    return {"models": [{"id": k, "api_id": m.id, "name": m.name, "params": m.params, "context_window": m.context,
                        "speed": m.speed, "pricing": {"input_per_1m": m.price_input, "output_per_1m": m.price_output},
                        "tier": m.tier, "rpm": m.rpm, "tpm": m.tpm, "rpd": m.rpd,
                        "supports_vision": m.supports_vision, "supports_tools": m.supports_tools}
                       for k, m in MODELS.items()], "total": len(MODELS)}

@app.get("/agents")
async def list_agents():
    agents = []
    for at, prompt in AGENT_PROMPTS.items():
        tiers = [tn for tn, cfg in TIER_CONFIGS.items() if at in cfg.available_agents]
        agents.append({"id": at.value, "name": at.value.replace("_", " ").title(),
                       "description": prompt[:100] + "...", "available_in": tiers})
    return {"agents": agents, "total": len(agents)}

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "6.0.0",
            "groq_api": "configured" if GROQ_API_KEY else "not set",
            "tavily_api": "configured" if TAVILY_API_KEY else "not set",
            "exa_api": "configured" if EXA_API_KEY else "not set",
            "mediastack_api": "configured" if MEDIASTACK_API_KEY else "not set",
            "models_loaded": len(MODELS), "agents_loaded": len(AGENT_PROMPTS),
            "tts_enabled": False,
            "programming_memory": True,
            "identity_locked": "ApoloXia · The Shield Technology · Panamá · Amelio Delgado",
            "note": "Optimizado para velocidad · Identidad forzada ApoloXia"}

@app.post("/search-web")
async def web_search(query: str, max_results: int = 5):
    tasks = []
    if TAVILY_API_KEY: tasks.append(search_tavily(query, max_results))
    if EXA_API_KEY: tasks.append(search_exa_ai(query, max_results))
    if MEDIASTACK_API_KEY: tasks.append(search_mediastack(query, max_results))
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_results = []
        for res in results:
            if isinstance(res, list):
                all_results.extend(res)
        return {"query": query, "results": all_results[:max_results*3], "count": len(all_results)}
    return {"query": query, "results": [], "count": 0}

@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str):
    user_mem = memory.memories.get(user_id, {})
    convs = [{"conversation_id": cid, "message_count": len(msgs),
              "last_access": memory.last_access.get(user_id, {}).get(cid).isoformat() if memory.last_access.get(user_id, {}).get(cid) else None}
             for cid, msgs in user_mem.items()]
    return {"user_id": user_id, "conversations": convs}

@app.delete("/conversations/{user_id}/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    if user_id in memory.memories and conversation_id in memory.memories[user_id]:
        del memory.memories[user_id][conversation_id]
        if conversation_id in memory.last_access.get(user_id, {}):
            del memory.last_access[user_id][conversation_id]
        return {"message": "Conversación eliminada"}
    raise HTTPException(404, "Conversación no encontrada")

@app.get("/memory/programming/{user_id}")
async def get_programming_memory(user_id: str):
    mem = memory.programming_memory.get(user_id)
    if not mem:
        return {"user_id": user_id, "memory": {}, "message": "Sin actividad registrada"}
    return {
        "user_id": user_id,
        "languages": dict(mem["languages"]),
        "frameworks": dict(mem["frameworks"]),
        "topics_history": mem["topics_history"],
        "preferred_style": mem["preferred_style"],
    }

# ============ CONFIGURACIÓN DE USUARIO ============
@app.get("/user-config/{user_id}")
async def get_user_config(user_id: str):
    tier = memory.get_user_tier(user_id)
    config = TIER_CONFIGS[tier]
    user_cfg = memory.get_user_config(user_id)
    return {
        "user_id": user_id, "tier": tier, "tier_name": config.name,
        "theme": user_cfg.get("theme", "dark"), "language": user_cfg.get("language", "es"),
        "notifications": user_cfg.get("notifications", True),
        "remaining_daily": memory.get_remaining(user_id, tier),
        "max_daily_responses": config.max_daily_responses,
        "available_models": [{"id": k, "name": MODELS[k].name, "params": MODELS[k].params, "context": MODELS[k].context,
                            "speed": MODELS[k].speed, "price_input": MODELS[k].price_input, "price_output": MODELS[k].price_output,
                            "rpm": MODELS[k].rpm, "tpm": MODELS[k].tpm, "rpd": MODELS[k].rpd,
                            "tier": MODELS[k].tier}
                           for k in config.available_models if k in MODELS],
        "available_agents": [a.value for a in config.available_agents],
        "features": {"web_search": config.supports_web_search, "file_upload": config.supports_file_upload,
                     "long_responses": config.supports_long_responses, "multi_agent": config.supports_multi_agent,
                     "code_execution": config.supports_code_execution, "deep_analysis": config.supports_deep_analysis,
                     "projects": config.supports_projects, "custom_gpts": config.supports_custom_gpts}
    }

@app.post("/user-config/{user_id}")
async def update_user_config(user_id: str, config: UserConfigUpdate):
    current = memory.get_user_config(user_id)
    if config.tier and config.tier in TIER_CONFIGS:
        memory.set_user_tier(user_id, config.tier)
    if config.theme: current["theme"] = config.theme
    if config.language: current["language"] = config.language
    if config.notifications is not None: current["notifications"] = config.notifications
    memory.set_user_config(user_id, current)
    return {"message": "Configuración actualizada", "user_id": user_id, "config": current}

# ============ ARCHIVOS ESTÁTICOS ============
def _get_file_path(filename: str) -> str:
    if os.path.exists(filename) and os.path.isfile(filename):
        return filename
    parent_path = os.path.join(os.path.dirname(__file__), "..", filename)
    if os.path.exists(parent_path) and os.path.isfile(parent_path):
        return parent_path
    return filename

@app.get("/")
async def serve_index():
    return FileResponse(_get_file_path("index.html"))

@app.get("/chat")
async def serve_chat():
    return FileResponse(_get_file_path("chat.html"))

@app.get("/{filename}")
async def serve_static_file(filename: str):
    api_routes = {"chat", "models", "agents", "health", "conversations", "tier-info",
                  "user-config", "upgrade-tier", "search-web", "share", "memory", "integrations"}
    if filename in api_routes:
        raise HTTPException(404, "Not found")
    if filename.startswith(".") or ".." in filename:
        raise HTTPException(403, "Forbidden")
    file_path = _get_file_path(filename)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    raise HTTPException(404, "Archivo no encontrado")

# ============ MAIN ============
if __name__ == "__main__":
    print("🚀 Iniciando ApoloXia Server v6.0.0 (Optimizado para velocidad)")
    print(f"📊 Modelos activos: {len(MODELS)} | 🤖 Agentes: {len(AGENT_PROMPTS)}")
    print("🧠 Identidad: ApoloXia · The Shield Technology · Panamá · Amelio Delgado")
    print("🔍 Motores de búsqueda:")
    print(f"   - Tavily: {'✅' if TAVILY_API_KEY else '❌'}")
    print(f"   - Exa AI: {'✅' if EXA_API_KEY else '❌'}")
    print(f"   - MediaStack: {'✅' if MEDIASTACK_API_KEY else '❌'}")
    print("🔇 Voz/TTS desactivada")
    print("⚡ Optimizaciones aplicadas:")
    print("   - max_tokens reducidos: 700/2500/5000")
    print("   - Historial más corto: 6/18/30")
    print("   - Timeouts: Groq 100s, búsquedas 10s")
    print("   - Solo 2 reintentos de modelo")
    print("   - Rate limiter al 90%")
    print("🔒 Identidad forzada: nunca dirá ChatGPT")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)