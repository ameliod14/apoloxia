from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.responses import StreamingResponse
import uvicorn
from datetime import datetime
import base64
import json
import aiohttp
from typing import Optional, List, Dict, Any
import asyncio
import hashlib
import time
import os
from pathlib import Path

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TAVILY_URL = "https://api.tavily.com/search"

# ==========================================
# 4 MODELOS GROQ 2026 - DISTRIBUCIÓN FREE/PRO
# ==========================================

MODELS = {
    "free": {
        "general": "llama-3.3-70b-versatile",
        "codigo": "codellama-34b-instruct"
    },
    "pro": {
        "general": "deepseek-r1-distill-llama-70b",
        "codigo": "qwen-2.5-coder-32b",
        "investigador": "llama-3.3-70b-versatile",
        "planificador": "deepseek-r1-distill-llama-70b",
        "analista": "llama-3.3-70b-versatile"
    }
}

# ==========================================
# CONFIGURACIÓN DE PLANES CON TODAS LAS FEATURES
# ==========================================

PLAN_CONFIG = {
    "free": {
        "name": "ApoloXia",
        "daily_messages": 100,
        "max_tokens": 2000,
        "context_window": "128K",
        "features": [
            "Respuestas extensas y bien definidas",
            "2 Agentes especializados",
            "Memoria de contexto completa",
            "Análisis profundo general",
            "Código optimizado",
            "Búsqueda web básica"
        ],
        "limitations": [
            "Sin SSO/MFA empresarial",
            "Sin integración SharePoint",
            "Sin transcripción de reuniones",
            "Sin exportación de documentos",
            "Sin agentes avanzados de investigación"
        ]
    },
    "pro": {
        "name": "ApoloXia Pro Enterprise",
        "daily_messages": float('inf'),
        "max_tokens": 8000,
        "context_window": "1M",
        "features": [
            "Protección SSO (Single Sign-On)",
            "Autenticación MFA (Multi-Factor)",
            "Protección de privacidad total",
            "Datos NUNCA usados para entrenamiento",
            "Encriptación end-to-end",
            "Comparte proyectos personalizados",
            "Comparte GPTs personalizados",
            "Integración con SharePoint",
            "Integración con Microsoft 365",
            "Integración con Google Workspace",
            "Gestión de equipos y permisos",
            "Facturación centralizada",
            "Gestión de usuarios empresarial",
            "Roles y permisos avanzados",
            "Auditoría de uso completa",
            "Transcripción automática de reuniones",
            "Notas de reuniones con IA",
            "Resúmenes ejecutivos automáticos",
            "Agente Codex (programación experta)",
            "Agente Investigador (análisis profundo)",
            "Agente Planificador (gestión de proyectos)",
            "Agente Analista (datos empresariales)",
            "Agent Swarm (coordinación multi-agente)",
            "Acceso anticipado a funciones experimentales",
            "Contexto completo máxima capacidad",
            "Memoria ilimitada para respuestas inteligentes",
            "Planificación avanzada de tareas",
            "Automatización de flujos de trabajo",
            "Proyectos y GPTs personalizados ilimitados",
            "Análisis predictivo avanzado"
        ]
    }
}

# ==========================================
# SISTEMA DE MEMORIA Y SESIONES
# ==========================================

class UserSession:
    def __init__(self, user_id: str, tier: str):
        self.user_id = user_id
        self.tier = tier
        self.message_count = 0
        self.conversations: Dict[str, List[Dict]] = {}
        self.created_at = time.time()
        self.last_active = time.time()
        self.total_tokens_used = 0
        
    def can_send_message(self) -> bool:
        if self.tier == "pro":
            return True
        return self.message_count < PLAN_CONFIG["free"]["daily_messages"]
    
    def add_message(self, conversation_id: str, role: str, content: str):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        if len(self.conversations[conversation_id]) > 50:
            self.conversations[conversation_id] = self.conversations[conversation_id][-50:]
        
        self.message_count += 1
        self.last_active = time.time()
    
    def get_context(self, conversation_id: str, max_messages: int = 10) -> List[Dict]:
        if conversation_id not in self.conversations:
            return []
        context_limit = 50 if self.tier == "pro" else max_messages
        return self.conversations[conversation_id][-context_limit:]

sessions: Dict[str, UserSession] = {}

def get_session(user_id: str, tier: str) -> UserSession:
    key = f"{user_id}_{tier}"
    if key not in sessions:
        sessions[key] = UserSession(user_id, tier)
    return sessions[key]

# ==========================================
# FUNCIONES DE BÚSQUEDA Y IA
# ==========================================

async def search_web(query: str, depth: str = "advanced") -> Dict[str, Any]:
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": depth,
                "include_answer": True,
                "include_images": False,
                "max_results": 8 if depth == "advanced" else 4
            }
            async with session.post(TAVILY_URL, json=payload, timeout=20) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "answer": data.get("answer", ""),
                        "results": data.get("results", []),
                        "query": query
                    }
                else:
                    return {"success": False, "error": f"Status {response.status}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

async def call_groq(
    messages: List[Dict], 
    model: str, 
    max_tokens: int = 2000,
    temperature: float = 0.7,
    stream: bool = False
) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROQ_URL, headers=headers, json=payload, timeout=45) as response:
                if response.status != 200:
                    error = await response.text()
                    print(f"❌ Groq error: {error}")
                    if model != "llama-3.3-70b-versatile":
                        print(f"🔄 Intentando fallback...")
                        return await call_groq(messages, "llama-3.3-70b-versatile", max_tokens, temperature)
                    return "Error del servicio. Intenta de nuevo en un momento."
                data = await response.json()
                if 'choices' in data and len(data['choices']) > 0:
                    return data['choices'][0]['message']['content']
                else:
                    return "No se pudo generar una respuesta adecuada."
    except asyncio.TimeoutError:
        return "La consulta está tomando demasiado tiempo. Intenta con una pregunta más específica."
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error de conexión. Verifica tu conexión e intenta de nuevo."

def build_system_prompt(
    tier: str, 
    agent: str, 
    has_web_info: bool = False,
    conversation_context: str = ""
) -> str:
    current_date = "Abril 2026"
    if tier == "free":
        base = f"""Eres Apolo, un asistente AI avanzado de clase mundial. Fecha actual: {current_date}.

🎯 TU OBJETIVO: Proporcionar respuestas EXTENSAS, BIEN ESTRUCTURADAS y COMPLETAS.
NUNCA seas breve. Siempre desarrolla tus respuestas con profundidad.

📏 REGLAS OBLIGATORIAS (Plan Free):
1. **EXTENSIÓN**: Mínimo 300-500 palabras. NUNCA menos.
2. **ESTRUCTURA**: Usa ## para títulos de secciones
3. **CONTENIDO**: 
   - Introducción contextual
   - Desarrollo con 3-5 puntos clave
   - Ejemplos prácticos o casos de uso
   - Conclusión con recomendaciones
4. **TONO**: Profesional pero accesible
5. **IDIOMA**: Español neutro, gramática perfecta
6. **FORMATO**: Markdown con negritas, listas y énfasis"""
    else:
        base = f"""Eres Apolo Pro, el asistente AI empresarial más avanzado. Fecha: {current_date}.

🏢 MODO ENTERPRISE ACTIVADO - Respuestas de NIVEL CONSULTORÍA

📏 REGLAS OBLIGATORIAS (Plan Pro):
1. **EXTENSIÓN**: Mínimo 800-1500 palabras. Respuestas exhaustivas.
2. **ESTRUCTURA PROFESIONAL**:
   ## Resumen Ejecutivo
   ## Análisis Detallado (5-7 secciones)
   ### Subtemas específicos
   ## Marco Teórico/Contextual
   ## Casos de Estudio/Ejemplos Prácticos
   ## Implicaciones Estratégicas
   ## Recomendaciones Accionables
   ## Conclusión y Próximos Pasos

3. **PROFUNDIDAD**: Análisis multidimensional, considera múltiples perspectivas
4. **DATOS**: Cuando aplica, incluye estadísticas, porcentajes, proyecciones
5. **FUENTES**: Menciona fuentes de autoridad (Harvard, McKinsey, etc.)
6. **TONO**: Ejecutivo, estratégico, orientado a resultados
7. **IDIOMA**: Español empresarial perfecto"""

    if has_web_info:
        base += f"\n\n🔍 INFORMACIÓN WEB ACTUALIZADA (2024-2026) disponible. Usa estos datos recientes para respuestas precisas."
    if conversation_context:
        base += f"\n\n💬 CONTEXTO DE LA CONVERSACIÓN:\n{conversation_context}"
    
    agent_specs = {
        "general": "",
        "codigo": """
💻 ESPECIALIDAD: Ingeniería de Software
- Explica algoritmos paso a paso con complejidad O(n)
- Proporciona código limpio, comentado y optimizado
- Incluye manejo de errores y casos edge
- Menciona mejores prácticas (SOLID, DRY, KISS)
- Da alternativas de implementación""",
        "investigador": """
🔬 ESPECIALIDAD: Investigación Académica y Análisis Profundo
- Metodología científica rigurosa
- Múltiples perspectivas teóricas
- Revisión de literatura actualizada
- Análisis crítico de fuentes
- Conclusiones basadas en evidencia
- Sugerencias para investigación futura""",
        "planificador": """
📋 ESPECIALIDAD: Gestión Estratégica de Proyectos
- Marco metodológico (PMI, Agile, Lean)
- Cronograma detallado con milestones
- Asignación de recursos (humanos, financieros, técnicos)
- Gestión de riesgos con matriz de probabilidad/impacto
- KPIs y métricas de éxito
- Plan de contingencia""",
        "analista": """
📊 ESPECIALIDAD: Análisis de Datos y Business Intelligence
- Análisis descriptivo, predictivo y prescriptivo
- Métricas clave (KPIs, OKRs)
- Segmentación y análisis de cohortes
- Proyecciones financieras/operativas
- Visualización de datos (conceptual)
- Insights accionables para stakeholders"""
    }
    if agent in agent_specs:
        base += agent_specs[agent]
    if tier == "pro":
        base += """
        
🔐 CAPACIDADES PRO ENTERPRISE:
- Análisis de seguridad y compliance (SSO, MFA, GDPR)
- Integración con ecosistemas empresariales
- Automatización de flujos de trabajo complejos
- Transcripción y análisis de reuniones
- Generación de documentos ejecutivos"""
    return base

def detect_web_search_need(message: str) -> tuple[bool, str]:
    message_lower = message.lower()
    urgent_keywords = [
        "noticia", "noticias", "hoy", "ayer", "última hora", "breaking",
        "acaba de", "acaban de", "acaba de pasar", "está pasando",
        "elecciones", "resultados", "ganó", "perdió", "votación",
        "guerra", "conflicto", "ataque", "crisis", "emergencia",
        "terremoto", "huracán", "inundación", "desastre",
        "precio del dólar", "precio del euro", "bitcoin", "acciones",
        "mercado hoy", "bolsa hoy", "wall street"
    ]
    standard_keywords = [
        "2024", "2025", "2026", "nuevo", "nueva", "último", "reciente",
        "microsoft", "google", "apple", "openai", "meta", "tesla",
        "inteligencia artificial", "ia", "gpt", "chatgpt", "claude",
        "ley", "nueva ley", "regulación", "gobierno", "presidente",
        "mundial", "olimpiadas", "champions", "super bowl"
    ]
    for kw in urgent_keywords:
        if kw in message_lower:
            return True, "advanced"
    for kw in standard_keywords:
        if kw in message_lower:
            return True, "basic"
    return False, "none"

# ==========================================
# ENDPOINTS API
# ==========================================

# ✅ Endpoint raíz: sirve index.html (login) primero
@app.get("/", response_class=HTMLResponse)
async def root():
    # Primero busca index.html (página de login)
    html_path = Path(__file__).parent.parent / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    # Si no existe, fallback a chat.html
    html_path = Path(__file__).parent.parent / "chat.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>ApoloXia</h1><p>Frontend no encontrado</p>")

# ✅ Endpoint /chat: sirve el chatbot
@app.get("/chat", response_class=HTMLResponse)
async def chat_page():
    html_path = Path(__file__).parent.parent / "chat.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>Chat no disponible</h1>")

# Los demás endpoints (health, config, user-status, chat post, etc.) permanecen igual

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_active": True,
        "web_search": True
    }

@app.get("/config")
def get_config():
    return {
        "plans": PLAN_CONFIG,
        "models": MODELS,
        "agents": {
            "free": ["general", "codigo"],
            "pro": ["general", "codigo", "investigador", "planificador", "analista"]
        }
    }

@app.get("/user-status")
def user_status(user_id: str, tier: str = "free"):
    session = get_session(user_id, tier)
    config = PLAN_CONFIG[tier]
    remaining = "unlimited" if config["daily_messages"] == float('inf') else max(0, config["daily_messages"] - session.message_count)
    return {
        "user_id": user_id,
        "tier": tier,
        "plan_name": config["name"],
        "remaining_messages": remaining,
        "message_count": session.message_count,
        "max_tokens": config["max_tokens"],
        "context_window": config["context_window"],
        "features": config["features"],
        "conversations_active": len(session.conversations)
    }

@app.post("/chat")
async def chat(
    message: str = Form(...),
    user_id: str = Form(...),
    tier: str = Form("free"),
    agent: str = Form("general"),
    conversation_id: str = Form("default"),
    file: Optional[UploadFile] = File(None)
):
    print(f"\n{'='*60}")
    print(f"📝 [{tier.upper()}] {agent.upper()} | {user_id[:15]}...")
    print(f"💬 Mensaje: {message[:70]}...")
    
    session = get_session(user_id, tier)
    config = PLAN_CONFIG[tier]
    
    if not session.can_send_message():
        return JSONResponse({
            "error": "limit_reached",
            "reply": f"""Has alcanzado el límite de {config['daily_messages']} mensajes diarios.

🚀 **Actualiza a ApoloXia Pro** para desbloquear:
• Mensajes ilimitados
• 5 Agentes especializados con modelos avanzados
• Contexto de 1M tokens (memoria masiva)
• Integración empresarial (SSO, MFA, SharePoint)
• Transcripción de reuniones
• Agente Codex para programación experta""",
            "upgrade_required": True,
            "remaining_messages": 0
        })
    
    file_data = None
    file_type = None
    if file:
        if tier != "pro":
            return JSONResponse({
                "error": "feature_not_available",
                "reply": "📎 La subida de archivos hasta 500MB requiere ApoloXia Pro Enterprise.",
                "upgrade_required": True
            })
        content = await file.read()
        if len(content) > 500 * 1024 * 1024:
            return JSONResponse({
                "error": "file_too_large",
                "reply": "Archivo demasiado grande. Límite: 500MB en Pro."
            })
        file_data = base64.b64encode(content).decode()
        file_type = file.filename.split('.')[-1].lower()
        print(f"📎 Archivo: {file.filename} ({len(content)/1024/1024:.2f} MB)")
    
    available_agents = list(MODELS[tier].keys())
    if agent not in available_agents:
        agent = "general"
    
    needs_web, search_depth = detect_web_search_need(message)
    web_info = {}
    if needs_web:
        print(f"🔍 Búsqueda web ({search_depth})...")
        web_info = await search_web(message, search_depth)
        if web_info.get("success"):
            print(f"✅ Web: {len(web_info.get('results', []))} resultados")
        else:
            print(f"⚠️ Web falló: {web_info.get('error', 'unknown')}")
    
    context_messages = session.get_context(conversation_id)
    context_str = ""
    if context_messages:
        recent = context_messages[-5:]
        context_str = "Conversación reciente:\n" + "\n".join([
            f"{'Usuario' if m['role'] == 'user' else 'Apolo'}: {m['content'][:100]}..."
            for m in recent
        ])
    
    system_prompt = build_system_prompt(
        tier=tier,
        agent=agent,
        has_web_info=web_info.get("success", False),
        conversation_context=context_str
    )
    
    user_content = message
    if web_info.get("success"):
        web_context = f"""INFORMACIÓN WEB ACTUALIZADA (2024-2026):
{web_info.get('answer', '')}

Fuentes principales:
"""
        for i, r in enumerate(web_info.get('results', [])[:3], 1):
            web_context += f"{i}. {r.get('title', '')}: {r.get('content', '')[:300]}...\n"
        user_content = f"{web_context}\n\nPREGUNTA DEL USUARIO:\n{message}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        *[{"role": m["role"], "content": m["content"]} for m in context_messages[-3:]],
        {"role": "user", "content": user_content}
    ]
    
    model = MODELS[tier].get(agent, MODELS[tier]["general"])
    max_tokens = config["max_tokens"]
    print(f"🤖 Modelo: {model}")
    print(f"🎯 Tokens máx: {max_tokens}")
    
    reply = await call_groq(messages, model, max_tokens, temperature=0.7)
    
    session.add_message(conversation_id, "user", message)
    session.add_message(conversation_id, "assistant", reply)
    
    remaining = "unlimited" if config["daily_messages"] == float('inf') else max(0, config["daily_messages"] - session.message_count)
    print(f"✅ Respuesta: {len(reply)} caracteres | Restantes: {remaining}")
    print(f"{'='*60}\n")
    
    return {
        "reply": reply,
        "file_data": file_data,
        "file_type": file_type,
        "remaining_messages": remaining,
        "agent_used": agent,
        "model_used": model,
        "tier": tier,
        "web_search_used": web_info.get("success", False),
        "conversation_id": conversation_id,
        "tokens_used": len(reply) // 4
    }

@app.post("/deep-research")
async def deep_research(
    topic: str = Form(...),
    user_id: str = Form(...),
    tier: str = Form("pro")
):
    if tier != "pro":
        return JSONResponse({
            "error": "pro_required",
            "message": "Deep Research requiere ApoloXia Pro Enterprise"
        }, status_code=403)
    
    web_info = await search_web(topic, "advanced")
    if not web_info.get("success"):
        return {"error": "search_failed", "message": "No se pudo realizar la investigación"}
    
    system_prompt = """Eres un investigador académico senior. Realiza un análisis exhaustivo con:
1. Estado del arte
2. Múltiples perspectivas teóricas
3. Evidencia empírica reciente
4. Análisis crítico de fuentes
5. Conclusiones y recomendaciones para investigación futura"""
    
    context = f"""TEMA A INVESTIGAR: {topic}

DATOS WEB RECOPILADOS:
{web_info.get('answer', '')}

FUENTES:
"""
    for r in web_info.get("results", []):
        context += f"- {r.get('title')}: {r.get('content', '')[:400]}...\n  URL: {r.get('url')}\n\n"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{context}\n\nGenera un informe de investigación completo en español."}
    ]
    reply = await call_groq(messages, "llama-3.3-70b-versatile", 8000, temperature=0.5)
    return {
        "research_report": reply,
        "sources": web_info.get("results", []),
        "topic": topic,
        "source_count": len(web_info.get("results", []))
    }

@app.post("/generate-document")
async def generate_document(
    content: str = Form(...),
    doc_type: str = Form("pdf"),
    user_id: str = Form(...),
    tier: str = Form("pro")
):
    if tier != "pro":
        return JSONResponse({"error": "pro_required"}, status_code=403)
    
    if doc_type == "pdf":
        formatted = f"""APOLOXIA PRO - DOCUMENTO GENERADO
{'='*50}

{content}

{'='*50}
Generado: {datetime.now().isoformat()}
Usuario: {user_id}
"""
    elif doc_type == "xlsx":
        formatted = f"Título,Contenido,Fecha\nApoloXia Export,{content[:100]}...,{datetime.now().isoformat()}"
    else:
        formatted = content
    
    encoded = base64.b64encode(formatted.encode()).decode()
    return {
        "file_data": encoded,
        "file_type": doc_type,
        "filename": f"apoloxia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{doc_type}"
    }

@app.get("/conversations/{user_id}")
def get_conversations(user_id: str, tier: str = "free"):
    session = get_session(user_id, tier)
    return {
        "conversations": {
            k: [{"role": m["role"], "content": m["content"][:200], "time": m["timestamp"]} 
                for m in v[-10:]]
            for k, v in session.conversations.items()
        }
    }

# ==========================================
# INICIO DEL SERVIDOR
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print("=" * 80)
    print("🚀 APOLOXIA PRO ENTERPRISE - API v3.0")
    print("=" * 80)
    print("\n🤖 MODELOS DISPONIBLES:")
    print(f"   FREE - General: {MODELS['free']['general']}")
    print(f"   FREE - Código:  {MODELS['free']['codigo']}")
    print(f"   PRO  - General: {MODELS['pro']['general']}")
    print(f"   PRO  - Código:  {MODELS['pro']['codigo']}")
    print(f"   PRO  - Investigador: {MODELS['pro']['investigador']}")
    print(f"   PRO  - Planificador: {MODELS['pro']['planificador']}")
    print(f"   PRO  - Analista:     {MODELS['pro']['analista']}")
    print(f"\n📍 Puerto: {port}")
    print("📖 Docs: http://localhost:8000/docs")
    print("🔧 Config: http://localhost:8000/config")
    print("=" * 80)
    print("⛔ Presiona CTRL+C para detener\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )