"""
ApoloXia - Configuración y Gestión de Planes
Define los límites y capacidades según el plan del usuario
"""

import os
from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum

# API Key de OpenRouter
OPENROUTER_API_KEY = "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"

class PlanType(Enum):
    FREE = "free"
    PRO = "pro"  # Apolo Plus

@dataclass
class PlanFeatures:
    """Características de cada plan"""
    name: str
    max_context_tokens: int
    memory_limit: int  # en mensajes
    allow_image_generation: bool
    allow_web_search: bool
    allow_agents: bool
    allow_file_upload: bool
    allow_sso: bool
    allow_advanced_models: bool
    max_response_length: str
    codex_access: bool
    transcription: bool
    sharepoint_integration: bool
    privacy_protection: bool
    
    # Límites de rate
    requests_per_minute: int
    max_concurrent_requests: int

# Configuración de planes
PLANS = {
    PlanType.FREE: PlanFeatures(
        name="Apolo Free",
        max_context_tokens=4096,
        memory_limit=10,  # Solo últimos 10 mensajes
        allow_image_generation=True,  # Prueba básica
        allow_web_search=False,
        allow_agents=False,
        allow_file_upload=False,
        allow_sso=False,
        allow_advanced_models=False,
        max_response_length="breve",
        codex_access=False,
        transcription=False,
        sharepoint_integration=False,
        privacy_protection=False,
        requests_per_minute=10,
        max_concurrent_requests=2
    ),
    PlanType.PRO: PlanFeatures(
        name="Apolo Plus",
        max_context_tokens=128000,  # Contexto completo
        memory_limit=1000,  # Memoria extensa
        allow_image_generation=True,
        allow_web_search=True,
        allow_agents=True,
        allow_file_upload=True,
        allow_sso=True,
        allow_advanced_models=True,
        max_response_length="detallada",
        codex_access=True,  # Asientos Codex disponibles
        transcription=True,  # Notas de reuniones
        sharepoint_integration=True,
        privacy_protection=True,  # Datos no usados para entrenamiento
        requests_per_minute=60,
        max_concurrent_requests=10
    )
}

# Modelos disponibles por plan
MODELS = {
    PlanType.FREE: {
        "chat": "openai/gpt-3.5-turbo",
        "vision": "openai/gpt-4o-mini",
        "image": "stabilityai/stable-diffusion-xl-base-1.0"
    },
    PlanType.PRO: {
        "chat": "openai/gpt-4o",
        "chat_advanced": "anthropic/claude-3-opus",
        "vision": "openai/gpt-4o",
        "image": "stabilityai/stable-diffusion-xl-base-1.0",
        "code": "openai/o1-mini",  # Codex
        "agent": "anthropic/claude-3-sonnet"
    }
}

class UserConfig:
    """Configuración del usuario actual"""
    def __init__(self, plan: PlanType = PlanType.FREE):
        self.plan = plan
        self.features = PLANS[plan]
        self.models = MODELS[plan]
        
    def upgrade_to_pro(self):
        """Upgrade a plan Pro"""
        self.plan = PlanType.PRO
        self.features = PLANS[PlanType.PRO]
        self.models = MODELS[PlanType.PRO]
        
    def can_use_feature(self, feature_name: str) -> bool:
        """Verifica si el usuario puede usar una característica"""
        return getattr(self.features, feature_name, False)