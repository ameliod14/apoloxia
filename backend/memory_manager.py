"""
ApoloXia - Gestión de Memoria y Contexto
Sistema de memoria inteligente con límites según plan
"""

import json
import time
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import hashlib

from config import UserConfig, PlanType

@dataclass
class MemoryEntry:
    """Entrada de memoria individual"""
    id: str
    content: str
    role: str  # user, assistant, system, file, web_result
    timestamp: float
    importance: float = 1.0  # 0.0 - 1.0 para priorización
    metadata: Dict[str, Any] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if not self.id:
            self.id = hashlib.md5(f"{self.content}{self.timestamp}".encode()).hexdigest()[:12]

class MemoryManager:
    """Gestor de memoria con límites según plan"""
    
    def __init__(self, user_config: UserConfig):
        self.config = user_config
        self.memories: List[MemoryEntry] = []
        self.session_context: Dict[str, Any] = {}
        self.current_session = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
    def add_memory(
        self,
        content: str,
        role: str = "user",
        importance: float = 1.0,
        metadata: Optional[Dict] = None
    ) -> MemoryEntry:
        """Añade una nueva memoria"""
        entry = MemoryEntry(
            id="",
            content=content,
            role=role,
            timestamp=time.time(),
            importance=importance,
            metadata=metadata or {},
            session_id=self.current_session
        )
        
        self.memories.append(entry)
        self._enforce_limits()
        
        return entry
        
    def _enforce_limits(self):
        """Aplica límites de memoria según el plan"""
        max_memories = self.config.features.memory_limit
        
        if len(self.memories) > max_memories:
            # Para Free: mantener solo los más recientes
            if self.config.plan == PlanType.FREE:
                self.memories = sorted(
                    self.memories,
                    key=lambda x: x.timestamp,
                    reverse=True
                )[:max_memories]
            else:
                # Para Pro: mantener por importancia + recencia
                self.memories = sorted(
                    self.memories,
                    key=lambda x: (x.importance * 0.7 + (x.timestamp / time.time()) * 0.3),
                    reverse=True
                )[:max_memories]
                
    def get_context(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Obtiene el contexto formateado para el modelo
        Respetando límites de tokens del plan
        """
        if limit is None:
            limit = self.config.features.memory_limit
            
        recent_memories = sorted(
            self.memories,
            key=lambda x: x.timestamp,
            reverse=True
        )[:limit]
        
        # Convertir a formato de mensajes
        context = []
        total_tokens = 0
        max_tokens = self.config.features.max_context_tokens
        
        for memory in reversed(recent_memories):  # Orden cronológico
            # Estimación simple de tokens (aprox 4 chars por token)
            estimated_tokens = len(memory.content) // 4
            
            if total_tokens + estimated_tokens > max_tokens * 0.8:  # 80% de margen
                break
                
            context.append({
                "role": memory.role,
                "content": memory.content
            })
            total_tokens += estimated_tokens
            
        return context
        
    def search_memories(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Búsqueda semántica simple en memoria (Pro)"""
        if self.config.plan == PlanType.FREE:
            return []
            
        # Búsqueda por similitud de palabras clave
        query_words = set(query.lower().split())
        scored = []
        
        for memory in self.memories:
            memory_words = set(memory.content.lower().split())
            score = len(query_words & memory_words) / len(query_words | memory_words)
            if score > 0.1:
                scored.append((score, memory))
                
        scored.sort(reverse=True)
        return [m for _, m in scored[:top_k]]
        
    def summarize_context(self) -> str:
        """Resume el contexto para mantener coherencia (Pro)"""
        if self.config.plan == PlanType.FREE:
            return ""
            
        # Crear resumen de la conversación
        topics = set()
        for m in self.memories:
            # Extracción simple de temas
            words = m.content.lower().split()[:5]
            topics.update(words)
            
        return f"Contexto actual: conversación sobre {', '.join(list(topics)[:10])}"
        
    def export_session(self) -> Dict:
        """Exporta la sesión actual (Pro)"""
        if not self.config.can_use_feature("allow_file_upload"):
            return {}
            
        return {
            "session_id": self.current_session,
            "plan": self.config.plan.value,
            "timestamp": time.time(),
            "memories": [asdict(m) for m in self.memories],
            "context_stats": {
                "total_memories": len(self.memories),
                "user_messages": len([m for m in self.memories if m.role == "user"]),
                "assistant_messages": len([m for m in self.memories if m.role == "assistant"])
            }
        }
        
    def clear_session(self):
        """Limpia la sesión actual"""
        self.memories = []
        self.session_context = {}
        self.current_session = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

class ContextCompressor:
    """Comprime contexto para mantener velocidad"""
    
    @staticmethod
    def compress_for_plan(context: List[Dict], plan: PlanType) -> List[Dict]:
        """Comprime el contexto según el plan"""
        if plan == PlanType.FREE:
            # Free: máximo 5 mensajes, truncados
            compressed = []
            for msg in context[-5:]:
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                compressed.append({
                    "role": msg["role"],
                    "content": content
                })
            return compressed
        else:
            # Pro: compresión inteligente
            return context