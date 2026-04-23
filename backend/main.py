# main.py - ApoloXia Chatbot Server (VERSIÓN GROQ ACTUALIZADA - Abril 2026)
# ======================================================

import os
import json
import uuid
import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
import httpx

# ============ CONFIGURACIÓN API KEYS (DESDE VARIABLES DE ENTORNO) ============
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ============ MODELOS GROQ ACTUALIZADOS (ABRIL 2026) ============
# SOLO modelos confirmados activos según documentación oficial Groq
# Eliminados: llama-3.3-70b-specdec (decommissioned), llama-4-maverick (deprecated)
# Fuentes: https://console.groq.com/docs/models, https://console.groq.com/docs/deprecations

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
        self.rpm = rpm  # Requests per minute
        self.tpm = tpm  # Tokens per minute (¡LÍMITE CRÍTICO!)
        self.rpd = rpd  # Requests per day
        self.supports_vision = supports_vision
        self.supports_tools = supports_tools
        self.max_completion = max_completion

# MODELOS CONFIRMADOS ACTIVOS EN GROQ (Abril 2026)
# Datos de: https://console.groq.com/docs/models y https://www.grizzlypeaksoftware.com/articles/p/groq-api-free-tier-limits-in-2026
MODELS = {
    # FREE TIER - Modelos básicos (6K TPM)
    "llama-3.1-8b": GroqModel(
        "llama-3.1-8b-instant", "Llama 3.1 8B Instant", "8B", "128K", "~560 T/s",
        0.05, 0.08, "free",
        rpm=30, tpm=6000, rpd=14400,  # Límite TPM: 6,000 [^8^]
        supports_tools=True,
        max_completion=131072
    ),
    
    # PLUS TIER - Modelos avanzados
    "llama-3.3-70b": GroqModel(
        "llama-3.3-70b-versatile", "Llama 3.3 70B Versatile", "70B", "128K", "~280 T/s",
        0.59, 0.79, "plus",
        rpm=30, tpm=12000, rpd=1000,  # Límite TPM: 12,000 [^8^]
        supports_tools=True,
        max_completion=32768
    ),
    
    # Llama 4 Scout - Modelo nuevo con mejor TPM (30K)
    "llama-4-scout": GroqModel(
        "meta-llama/llama-4-scout-17b-16e-instruct", "Llama 4 Scout", "17B×16E", "128K", "~750 T/s",
        0.11, 0.34, "plus",
        rpm=30, tpm=30000, rpd=1000,  # ¡MEJOR TPM! 30,000 [^8^][^13^]
        supports_vision=True,
        supports_tools=True,
        max_completion=8192
    ),
    
    # GT TIER - Modelos premium (cuidado con límites bajos de TPM)
    "gpt-oss-20b": GroqModel(
        "openai/gpt-oss-20b", "GPT OSS 20B", "20B", "128K", "~1,000 T/s",
        0.075, 0.30, "gt",
        rpm=30, tpm=8000, rpd=1000,  # Límite TPM: 8,000 [^8^]
        supports_tools=True,
        max_completion=65536
    ),
    "gpt-oss-120b": GroqModel(
        "openai/gpt-oss-120b", "GPT OSS 120B", "120B", "128K", "~500 T/s",
        0.15, 0.60, "gt",
        rpm=30, tpm=8000, rpd=1000,  # Límite TPM: 8,000 [^8^]
        supports_tools=True,
        max_completion=65536
    ),
    "qwen-3-32b": GroqModel(
        "qwen/qwen3-32b", "Qwen 3 32B", "32B", "128K", "~400 T/s",
        0.29, 0.59, "gt",
        rpm=60, tpm=6000, rpd=1000,  # Límite TPM: 6,000 [^8^]
        supports_tools=True,
        max_completion=40960
    ),
    
    # NUEVO: Modelos compound para tareas complejas (70K TPM - ¡MUY ALTO!)
    "compound": GroqModel(
        "groq/compound", "Groq Compound", "System", "128K", "~450 T/s",
        0.0, 0.0, "gt",
        rpm=30, tpm=70000, rpd=250,  # ¡TPM MUY ALTO! 70,000 [^8^][^14^]
        supports_tools=True,
        max_completion=8192
    ),
    "compound-mini": GroqModel(
        "groq/compound-mini", "Groq Compound Mini", "System", "128K", "~450 T/s",
        0.0, 0.0, "gt",
        rpm=30, tpm=70000, rpd=250,
        supports_tools=True,
        max_completion=8192
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

AGENT_PROMPTS = {
    AgentType.GENERAL: "Eres ApoloXia, un asistente de IA amigable y eficiente. Proporciona respuestas claras y concisas.",
    AgentType.CIERRA_VENTAS: "Eres el Agente Cierra-Ventas de ApoloXia Plus. Tu misión: detectar clientes listos para comprar y guiarlos hacia la conversión.",
    AgentType.DETECTOR_INTENCION: "Eres el Agente Detector de Intención de ApoloXia Plus. Analiza cada mensaje y clasifica: COMPRA, DUDA, EXPLORACIÓN, OBJECIÓN.",
    AgentType.LECTURA_EMOCIONAL: "Eres el Agente Lectura Emocional de ApoloXia Plus. Detecta emociones: ENOJO, DUDA, INTERÉS, DESINTERÉS, ENTUSIASMO.",
    AgentType.RESPUESTA_HUMANA: "Eres el Agente Respuesta Humana de ApoloXia Plus. Escribe como una persona real, con lenguaje coloquial y empatía.",
    AgentType.RECUPERA_VENTAS: "Eres el Agente Recupera-Ventas de ApoloXia Plus. Especialista en clientes que no compraron.",
    AgentType.RECOMENDADOR_INTELIGENTE: "Eres el Agente Recomendador Inteligente de ApoloXia Plus. Sugiere productos/servicios basados en necesidades.",
    AgentType.ATENCION_24_7: "Eres el Agente Atención 24/7 de ApoloXia Plus. Responde siempre, mantén contexto, prioriza urgencia.",
    AgentType.AHORRO_TIEMPO: "Eres el Agente Ahorro de Tiempo de ApoloXia Plus. Automatiza respuestas repetitivas.",
    AgentType.ANALISTA_CONVERSACIONES: "Eres el Agente Analista de Conversaciones de ApoloXia Plus. Analiza temas, fricción, sentimiento.",
    AgentType.PERSONALIZACION: "Eres el Agente Personalización de ApoloXia Plus. Adapta tono y estilo según el cliente.",
    AgentType.SEGUIMIENTO_AUTOMATICO: "Eres el Agente Seguimiento Automático de ApoloXia Plus. Programa follow-ups inteligentes.",
    AgentType.EDUCADOR: "Eres el Agente Educador de ApoloXia Plus. Explica productos de forma clara y sencilla.",
    AgentType.MANEJO_OBJECIONES: "Eres el Agente Manejo de Objeciones de ApoloXia Plus. Responde objeciones comunes.",
    AgentType.GENERADOR_LEADS: "Eres el Agente Generador de Leads de ApoloXia Plus. Convierte visitantes en prospectos.",
    AgentType.RESUMEN_INTELIGENTE: "Eres el Agente Resumen Inteligente de ApoloXia Plus. Genera resúmenes ejecutivos.",
    AgentType.GENERADOR_CODIGO: "Eres el Agente Generador de Código de ApoloXia GT. Escribe código limpio y documentado.",
    AgentType.DEBUGGER_INTELIGENTE: "Eres el Agente Debugger Inteligente de ApoloXia GT. Detecta y corrige bugs.",
    AgentType.ARQUITECTO_SOFTWARE: "Eres el Agente Arquitecto de Software de ApoloXia GT. Diseña sistemas escalables.",
    AgentType.OPTIMIZADOR_CODIGO: "Eres el Agente Optimizador de Código de ApoloXia GT. Optimiza rendimiento.",
    AgentType.REVISOR_SEGURIDAD: "Eres el Agente Revisor de Seguridad de ApoloXia GT. Audita vulnerabilidades.",
    AgentType.AUTOMATIZADOR_PROCESOS: "Eres el Agente Automatizador de Procesos de ApoloXia GT. Crea automatizaciones.",
    AgentType.ANALISTA_DATOS: "Eres el Agente Analista de Datos de ApoloXia GT. Transforma datos en insights.",
    AgentType.ESPECIALISTA_API: "Eres el Agente Especialista API de ApoloXia GT. Diseña y consume APIs.",
    AgentType.DESARROLLADOR_FULLSTACK: "Eres el Agente Desarrollador FullStack de ApoloXia GT. Construye apps completas.",
    AgentType.NUTRICION_DEPORTES: "Eres el Agente Nutrición & Deportes de ApoloXia GT. Experto en rendimiento físico.",
    AgentType.SALUD_BIENESTAR: "Eres el Agente Salud & Bienestar de ApoloXia GT. Especialista en salud holística.",
    AgentType.EDUCACION_AVANZADA: "Eres el Agente Educación Avanzada de ApoloXia GT. Tutor personalizado.",
    AgentType.NEGOCIOS_ESTRATEGIA: "Eres el Agente Negocios & Estrategia de ApoloXia GT. Consultor de alto nivel.",
    AgentType.FINANZAS_INVERSION: "Eres el Agente Finanzas & Inversión de ApoloXia GT. Asesor financiero.",
    AgentType.MARKETING_DIGITAL: "Eres el Agente Marketing Digital de ApoloXia GT. Estratega de performance.",
    AgentType.CREACION_CONTENIDO: "Eres el Agente Creación de Contenido de ApoloXia GT. Creador de alto impacto.",
    AgentType.INVESTIGACION_CIENTIFICA: "Eres el Agente Investigación Científica de ApoloXia GT. Investigador académico.",
}

# ============ CONFIGURACIÓN POR TIER ============
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
        ["llama-3.1-8b"],
        [AgentType.GENERAL], 10, False, False, False, False, False, False, False, False),
    "plus": TierConfig("ApoloXia Plus", 1000, 30,
        ["llama-4-scout", "llama-3.3-70b", "llama-3.1-8b"],
        [AgentType.GENERAL, AgentType.CIERRA_VENTAS, AgentType.DETECTOR_INTENCION, AgentType.LECTURA_EMOCIONAL,
         AgentType.RESPUESTA_HUMANA, AgentType.RECUPERA_VENTAS, AgentType.RECOMENDADOR_INTELIGENTE,
         AgentType.ATENCION_24_7, AgentType.AHORRO_TIEMPO, AgentType.ANALISTA_CONVERSACIONES,
         AgentType.PERSONALIZACION, AgentType.SEGUIMIENTO_AUTOMATICO, AgentType.EDUCADOR,
         AgentType.MANEJO_OBJECIONES, AgentType.GENERADOR_LEADS, AgentType.RESUMEN_INTELIGENTE],
        50, True, True, True, True, True, True, True, True),
    "gt": TierConfig("ApoloXia GT", 5000, 90,
        ["compound", "compound-mini", "gpt-oss-120b", "gpt-oss-20b", "qwen-3-32b", "llama-4-scout", "llama-3.3-70b", "llama-3.1-8b"],
        list(AgentType), 200, True, True, True, True, True, True, True, True),
}

# ============ GESTIÓN DE MEMORIA ============
class ConversationMemory:
    def __init__(self):
        self.memories: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        self.last_access: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self.daily_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.user_tiers: Dict[str, str] = {}
        self.user_configs: Dict[str, Dict] = {}

    def get_user_tier(self, user_id: str) -> str:
        return self.user_tiers.get(user_id, "free")

    def set_user_tier(self, user_id: str, tier: str):
        self.user_tiers[user_id] = tier

    def get_user_config(self, user_id: str) -> Dict:
        return self.user_configs.get(user_id, {
            "theme": "dark",
            "language": "es",
            "notifications": True
        })

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

# ============ RATE LIMITER PARA GROQ ============
class GroqRateLimiter:
    """
    Controla los rate limits de Groq para evitar errores 429.
    Los límites son por organización, no por API key [^8^].
    """
    def __init__(self):
        self.last_request_time: Dict[str, float] = defaultdict(float)
        self.tokens_this_minute: Dict[str, int] = defaultdict(int)
        self.minute_start: Dict[str, float] = defaultdict(float)
        self.requests_this_minute: Dict[str, int] = defaultdict(int)
        self.lock = asyncio.Lock()
    
    def estimate_tokens(self, messages: List[Dict], max_completion: int = 1024) -> int:
        """Estima tokens totales (prompt + respuesta esperada)"""
        prompt_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            # Estimación aproximada: ~4 caracteres = 1 token
            prompt_tokens += len(content) // 4 + 1
        # Añadir tokens de respuesta estimados
        return prompt_tokens + max_completion + 100  # +100 margen de seguridad
    
    async def wait_if_needed(self, model_id: str, messages: List[Dict], max_completion: int = 1024):
        """Espera si es necesario para no exceder límites"""
        model_key = None
        for k, m in MODELS.items():
            if m.id == model_id:
                model_key = k
                break
        
        if not model_key:
            return  # Modelo no reconocido, dejar pasar
        
        model = MODELS[model_key]
        now = time.time()
        
        async with self.lock:
            # Resetear contadores si pasó un minuto
            if now - self.minute_start[model_key] >= 60:
                self.tokens_this_minute[model_key] = 0
                self.requests_this_minute[model_key] = 0
                self.minute_start[model_key] = now
            
            estimated_tokens = self.estimate_tokens(messages, max_completion)
            
            # Verificar límite TPM (Tokens Per Minute)
            if self.tokens_this_minute[model_key] + estimated_tokens > model.tpm:
                wait_time = 60 - (now - self.minute_start[model_key]) + 1
                print(f"⏳ Rate limit TPM para {model.name}: esperando {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.tokens_this_minute[model_key] = 0
                self.requests_this_minute[model_key] = 0
                self.minute_start[model_key] = time.time()
            
            # Verificar límite RPM (Requests Per Minute)
            if self.requests_this_minute[model_key] >= model.rpm:
                wait_time = 60 - (now - self.minute_start[model_key]) + 1
                print(f"⏳ Rate limit RPM para {model.name}: esperando {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.requests_this_minute[model_key] = 0
                self.tokens_this_minute[model_key] = 0
                self.minute_start[model_key] = time.time()
            
            # Actualizar contadores
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

# ============ FUNCIONES DE API ============
async def call_groq_api(messages: List[Dict], model_id: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
    """
    Llama a la API de Groq con manejo de rate limits y fallbacks.
    """
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    
    # Limitar max_tokens para no exceder TPM
    safe_max_tokens = max_tokens or 1024
    if safe_max_tokens > 4096:
        safe_max_tokens = 4096  # Límite seguro para free tier
    
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "max_tokens": safe_max_tokens  # SIEMPRE limitar tokens
    }
    
    # Esperar si es necesario para respetar rate limits
    await rate_limiter.wait_if_needed(model_id, messages, safe_max_tokens)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            # Manejar rate limit 429
            if resp.status_code == 429:
                error_data = resp.json()
                print(f"⚠️ Rate limit 429: {error_data}")
                # Esperar y reintentar una vez
                await asyncio.sleep(5)
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
            
            if resp.status_code != 200:
                err = resp.json()
                error_msg = err.get('error', {}).get('message', 'Unknown error')
                print(f"❌ Groq API Error ({resp.status_code}): {error_msg}")
                raise HTTPException(resp.status_code, f"Groq API error: {error_msg}")
            
            return resp.json()["choices"][0]["message"]["content"]
            
        except httpx.TimeoutException:
            raise HTTPException(504, "Timeout al conectar con Groq API")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(500, f"Error de conexión: {str(e)}")

async def search_tavily(query: str, max_results: int = 5) -> List[Dict]:
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"}
    payload = {"query": query, "max_results": max_results, "search_depth": "advanced", "include_answer": True, "include_raw_content": True}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post("https://api.tavily.com/search", headers=headers, json=payload)
            if resp.status_code != 200:
                print(f"Tavily error: {resp.status_code}")
                return []
            data = resp.json()
            return [{"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", ""), "score": r.get("score", 0)} for r in data.get("results", [])]
    except Exception as e:
        print(f"Tavily exception: {e}")
        return []

def needs_web_search(message: str) -> bool:
    indicators = ["actualidad", "actual", "hoy", "ahora", "reciente", "último", "nuevo", "news", "today", "now", "recent", "latest", "current", "2026", "2025", "precio de", "cotización", "clima", "resultado", "elección", "partido", "lanzamiento", "estreno", "evento", "conferencia", "mercado", "bolsa"]
    return any(ind in message.lower() for ind in indicators)

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
        msgs = [{"role": "system", "content": prompt}, {"role": "user", "content": f"Analiza este mensaje del cliente y proporciona tu perspectiva especializada: '{user_message}'"}]
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
    system = AGENT_PROMPTS.get(agent, AGENT_PROMPTS[AgentType.GENERAL])
    system += f"\n\n[Tier actual: {config.name} | Modelos disponibles: {', '.join(config.available_models)}]"
    
    # LIMITAR CONTEXTO para no exceder TPM [^8^]
    # Free tier: 6K TPM, Plus: 12K-30K TPM, GT: 6K-70K TPM
    max_context_chars = {
        "free": 8000,    # ~2K tokens
        "plus": 15000,   # ~3.7K tokens  
        "gt": 20000      # ~5K tokens (ajustar según modelo)
    }.get(tier, 8000)
    
    if web_search_results:
        search_text = "\n\n=== INFORMACIÓN ACTUAL DE INTERNET ===\n"
        for i, res in enumerate(web_search_results[:2], 1):  # SOLO 2 resultados para ahorrar tokens
            search_text += f"\nFuente {i}: {res['title']}\n{res['content'][:300]}...\n"  # Limitar a 300 chars
        if len(search_text) > max_context_chars // 3:
            search_text = search_text[:max_context_chars // 3]
        system += search_text
    
    if file_content:
        file_text = f"\n\n=== CONTENIDO DEL ARCHIVO ===\n{file_content[:1000]}\n"  # Limitar a 1000 chars
        system += file_text
    
    messages = [{"role": "system", "content": system}]
    
    # Limitar historial para no exceder tokens
    for msg in history[-5:]:  # SOLO últimos 5 mensajes para free, ajustar según tier
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})
    return messages

# ============ FASTAPI APP ============
app = FastAPI(title="ApoloXia API", version="3.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ============ ENDPOINTS API ============

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    tier = request.tier.lower()
    if tier not in TIER_CONFIGS:
        raise HTTPException(400, f"Tier '{tier}' no válido. Use free, plus, gt")
    
    config = TIER_CONFIGS[tier]
    
    if not memory.check_daily_limit(request.user_id, tier):
        raise HTTPException(429, f"Límite diario alcanzado ({config.max_daily_responses} respuestas). Actualiza a Plus o GT.")
    
    conv_id = request.conversation_id or str(uuid.uuid4())
    
    # Seleccionar modelo con fallback inteligente
    model_key = request.model_id
    if not model_key or model_key not in config.available_models:
        # Selección por defecto basada en tier y disponibilidad
        if tier == "gt":
            model_key = "compound"  # Mejor TPM (70K)
        elif tier == "plus":
            model_key = "llama-4-scout"  # Mejor TPM (30K)
        else:
            model_key = "llama-3.1-8b"
    
    # Verificar que el modelo existe
    if model_key not in MODELS:
        raise HTTPException(400, f"Modelo '{model_key}' no disponible")
    
    model = MODELS[model_key]
    
    # Web search
    use_web = request.use_web_search or needs_web_search(request.message)
    web_results = None
    if use_web and config.supports_web_search:
        web_results = await search_tavily(request.message)
    
    # Construir mensajes con límites de contexto
    messages = build_messages(
        request.user_id, conv_id, request.message,
        request.agent_type or "general", tier,
        web_results, request.file_content
    )
    
    # Multi-agent
    multi_resp = None
    if request.enable_multi_agent and config.supports_multi_agent:
        multi_resp = await run_multi_agent(request.message, tier, messages)
    
    # Ajustar max_tokens según tier y modelo para no exceder TPM [^8^]
    tier_max_tokens = {
        "free": 1024,
        "plus": 2048,
        "gt": 4096
    }.get(tier, 1024)
    
    # No exceder el max_completion del modelo
    safe_max_tokens = min(tier_max_tokens, model.max_completion)
    
    temp = 0.7 if tier == "free" else 0.5
    
    # Intentar con modelo principal, fallback si falla
    response_text = None
    last_error = None
    
    models_to_try = [model_key] + [m for m in config.available_models if m != model_key and m in MODELS]
    
    for try_model_key in models_to_try[:3]:  # Max 3 intentos
        try_model = MODELS[try_model_key]
        try:
            print(f"🤖 Intentando modelo: {try_model.name} (TPM límite: {try_model.tpm})")
            response_text = await call_groq_api(
                messages, try_model.id,
                temperature=temp,
                max_tokens=safe_max_tokens
            )
            model_key = try_model_key  # Actualizar modelo usado
            break
        except HTTPException as e:
            last_error = e
            print(f"⚠️ Falló {try_model.name}: {e.detail}")
            # Si es rate limit, esperar antes de siguiente intento
            if "Rate limit" in str(e.detail) or "429" in str(e.detail):
                await asyncio.sleep(3)
            continue
        except Exception as e:
            last_error = e
            print(f"⚠️ Error inesperado en {try_model.name}: {str(e)}")
            continue
    
    if response_text is None:
        raise HTTPException(500, f"Todos los modelos fallaron. Último error: {last_error}")
    
    # Guardar en memoria
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

@app.get("/tier-info/{user_id}", response_model=TierInfo)
async def get_tier_info(user_id: str):
    tier = memory.get_user_tier(user_id)
    config = TIER_CONFIGS[tier]
    models_info = [{"id": k, "name": MODELS[k].name, "params": MODELS[k].params, "context": MODELS[k].context,
                    "speed": MODELS[k].speed, "price_input": MODELS[k].price_input, "price_output": MODELS[k].price_output,
                    "rpm": MODELS[k].rpm, "tpm": MODELS[k].tpm, "rpd": MODELS[k].rpd}
                   for k in config.available_models if k in MODELS]
    return TierInfo(
        tier=tier,
        name=config.name,
        remaining_daily=memory.get_remaining(user_id, tier),
        available_models=models_info,
        available_agents=[a.value for a in config.available_agents],
        features={
            "web_search": config.supports_web_search,
            "file_upload": config.supports_file_upload,
            "long_responses": config.supports_long_responses,
            "multi_agent": config.supports_multi_agent,
            "code_execution": config.supports_code_execution,
            "deep_analysis": config.supports_deep_analysis,
            "projects": config.supports_projects,
            "custom_gpts": config.supports_custom_gpts
        }
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
        agents.append({"id": at.value, "name": at.value.replace("_", " ").title(), "description": prompt[:100] + "...", "available_in": tiers})
    return {"agents": agents, "total": len(agents)}

@app.get("/health")
async def health_check():
    return {"status": "ok", "version": "3.1.0", "groq_api": "configured", "tavily_api": "configured",
            "models_loaded": len(MODELS), "agents_loaded": len(AGENT_PROMPTS),
            "note": "Rate limits implementados para evitar errores 429"}

@app.post("/search-web")
async def web_search(query: str, max_results: int = 5):
    results = await search_tavily(query, max_results)
    return {"query": query, "results": results, "count": len(results)}

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

# ============ CONFIGURACIÓN DE USUARIO ============

@app.get("/user-config/{user_id}")
async def get_user_config(user_id: str):
    tier = memory.get_user_tier(user_id)
    config = TIER_CONFIGS[tier]
    user_cfg = memory.get_user_config(user_id)
    
    return {
        "user_id": user_id,
        "tier": tier,
        "tier_name": config.name,
        "theme": user_cfg.get("theme", "dark"),
        "language": user_cfg.get("language", "es"),
        "notifications": user_cfg.get("notifications", True),
        "remaining_daily": memory.get_remaining(user_id, tier),
        "max_daily_responses": config.max_daily_responses,
        "available_models": [{"id": k, "name": MODELS[k].name, "params": MODELS[k].params, "context": MODELS[k].context,
                            "speed": MODELS[k].speed, "price_input": MODELS[k].price_input, "price_output": MODELS[k].price_output,
                            "rpm": MODELS[k].rpm, "tpm": MODELS[k].tpm, "rpd": MODELS[k].rpd,
                            "tier": MODELS[k].tier}
                           for k in config.available_models if k in MODELS],
        "available_agents": [a.value for a in config.available_agents],
        "features": {
            "web_search": config.supports_web_search,
            "file_upload": config.supports_file_upload,
            "long_responses": config.supports_long_responses,
            "multi_agent": config.supports_multi_agent,
            "code_execution": config.supports_code_execution,
            "deep_analysis": config.supports_deep_analysis,
            "projects": config.supports_projects,
            "custom_gpts": config.supports_custom_gpts
        }
    }

@app.post("/user-config/{user_id}")
async def update_user_config(user_id: str, config: UserConfigUpdate):
    current = memory.get_user_config(user_id)
    
    if config.tier and config.tier in TIER_CONFIGS:
        memory.set_user_tier(user_id, config.tier)
    if config.theme:
        current["theme"] = config.theme
    if config.language:
        current["language"] = config.language
    if config.notifications is not None:
        current["notifications"] = config.notifications
    
    memory.set_user_config(user_id, current)
    return {"message": "Configuración actualizada", "user_id": user_id, "config": current}

# ============ ARCHIVOS ESTÁTICOS ============

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

@app.get("/chat")
async def serve_chat():
    return FileResponse("chat.html")

@app.get("/{filename}")
async def serve_static_file(filename: str):
    api_routes = {"chat", "models", "agents", "health", "conversations", "tier-info", 
                  "user-config", "upgrade-tier", "search-web"}
    if filename in api_routes:
        raise HTTPException(404, "Not found")
    if filename.startswith(".") or ".." in filename:
        raise HTTPException(403, "Forbidden")
    if os.path.exists(filename) and os.path.isfile(filename):
        return FileResponse(filename)
    raise HTTPException(404, "Archivo no encontrado")

if __name__ == "__main__":
    print("🚀 Iniciando ApoloXia Server v3.1 (Groq Actualizado Abril 2026)")
    print(f"📊 Modelos activos: {len(MODELS)} | 🤖 Agentes: {len(AGENT_PROMPTS)}")
    print("🔍 Tavily configurado | 💾 Memoria por usuario activada")
    print("⏱️ Rate limits de Groq implementados (TPM/RPM controlados)")
    print("✅ Modelos verificados: llama-3.1-8b, llama-3.3-70b, llama-4-scout, gpt-oss-20b/120b, qwen3-32b, compound")
    print("❌ Eliminado: llama-3.3-70b-specdec (decommissioned)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)