"""
ApoloXia - Modelos de IA
Gestiona todas las llamadas a OpenRouter con optimización de velocidad
"""

import asyncio
import aiohttp
import json
from typing import AsyncGenerator, Dict, List, Optional, Any
from dataclasses import dataclass
import time

from config import OPENROUTER_API_KEY, UserConfig, PlanType

@dataclass
class Message:
    role: str  # system, user, assistant
    content: str
    timestamp: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class OpenRouterClient:
    """Cliente optimizado para OpenRouter"""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, user_config: UserConfig):
        self.config = user_config
        self.headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://apoloxia.com",
            "X-Title": "ApoloXia AI"
        }
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(user_config.features.max_concurrent_requests)
        
    async def __aenter__(self):
        timeout = aiohttp.ClientTimeout(total=60, connect=5)
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=self.headers
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        Genera respuestas de chat con streaming para máxima velocidad
        """
        async with self._semaphore:  # Control de concurrencia
            selected_model = model or self.config.models.get("chat")
            
            # Ajustar tokens según plan
            if max_tokens is None:
                max_tokens = 150 if self.config.plan == PlanType.FREE else 4000
            
            payload = {
                "model": selected_model,
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            try:
                async with self.session.post(
                    f"{self.BASE_URL}/chat/completions",
                    json=payload
                ) as response:
                    if stream:
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if line.startswith('data: '):
                                data = line[6:]
                                if data == '[DONE]':
                                    break
                                try:
                                    chunk = json.loads(data)
                                    if 'choices' in chunk and len(chunk['choices']) > 0:
                                        delta = chunk['choices'][0].get('delta', {})
                                        if 'content' in delta:
                                            yield delta['content']
                                except json.JSONDecodeError:
                                    continue
                    else:
                        data = await response.json()
                        yield data['choices'][0]['message']['content']
                        
            except Exception as e:
                yield f"Error: {str(e)}"
                
    async def generate_image(
        self,
        prompt: str,
        size: str = "1024x1024"
    ) -> Optional[str]:
        """Genera imágenes (disponible en ambos planes)"""
        if not self.config.can_use_feature("allow_image_generation"):
            return None
            
        model = self.config.models.get("image", "stabilityai/stable-diffusion-xl-base-1.0")
        
        payload = {
            "model": model,
            "prompt": prompt,
            "size": size
        }
        
        try:
            async with self.session.post(
                f"{self.BASE_URL}/images/generations",
                json=payload
            ) as response:
                data = await response.json()
                return data.get('data', [{}])[0].get('url')
        except Exception:
            return None
            
    async def web_search(
        self,
        query: str
    ) -> List[Dict[str, Any]]:
        """Búsqueda web para información actual (solo Pro)"""
        if not self.config.can_use_feature("allow_web_search"):
            return []
            
        # Usar modelo con capacidad de búsqueda
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente de investigación. Busca información actualizada y proporciona fuentes."
            },
            {
                "role": "user",
                "content": f"Busca información actualizada sobre: {query}. Proporciona resultados con fuentes."
            }
        ]
        
        # Simulación de búsqueda - en producción usar tool calling
        search_prompt = f"""Realiza una búsqueda web sobre: {query}
        Devuelve los resultados en formato JSON con: title, url, snippet"""
        
        results = []
        async for chunk in self.chat_completion(
            messages=[{"role": "user", "content": search_prompt}],
            model="perplexity/llama-3.1-sonar-small-128k-online",
            stream=False
        ):
            try:
                # Parsear resultados simulados
                results.append({
                    "title": f"Resultado sobre {query}",
                    "url": "https://search.example.com",
                    "snippet": chunk[:200]
                })
            except:
                pass
                
        return results
        
    async def code_analysis(
        self,
        code: str,
        language: str = "python"
    ) -> AsyncGenerator[str, None]:
        """Análisis de código con Codex (solo Pro)"""
        if not self.config.can_use_feature("codex_access"):
            yield "Función disponible solo en Apolo Plus"
            return
            
        model = self.config.models.get("code", "openai/o1-mini")
        
        messages = [
            {
                "role": "system",
                "content": f"Eres un experto en {language}. Analiza el código, optimízalo y explica los algoritmos."
            },
            {
                "role": "user",
                "content": f"Analiza este código:\n\n```{language}\n{code}\n```"
            }
        ]
        
        async for chunk in self.chat_completion(messages, model=model, temperature=0.3):
            yield chunk

class ResponseOptimizer:
    """Optimiza respuestas según el plan"""
    
    @staticmethod
    def optimize_for_plan(text: str, plan: PlanType) -> str:
        """Ajusta la respuesta según el plan del usuario"""
        if plan == PlanType.FREE:
            # Respuestas breves y sencillas
            sentences = text.split('. ')
            if len(sentences) > 3:
                return '. '.join(sentences[:3]) + '.'
            return text
        else:
            # Respuestas detalladas para Pro
            return text
            
    @staticmethod
    def add_pro_features(text: str, features_used: List[str]) -> str:
        """Añade metadatos para usuarios Pro"""
        if features_used:
            footer = f"\n\n---\n🔧 Características utilizadas: {', '.join(features_used)}"
            return text + footer
        return text