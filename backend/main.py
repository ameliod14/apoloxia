"""
ApoloXia - Bot de IA Principal
Sistema modular con planes Free y Pro (Apolo Plus)
"""

import asyncio
import sys
from typing import Optional, AsyncGenerator, List, Dict
import argparse

# Importar módulos locales
from config import UserConfig, PlanType, PLANS
from models import OpenRouterClient, ResponseOptimizer, Message
from memory_manager import MemoryManager, ContextCompressor

class ApoloXia:
    """Bot de IA ApoloXia - Free y Pro"""
    
    def __init__(self, plan: PlanType = PlanType.FREE):
        self.config = UserConfig(plan)
        self.memory = MemoryManager(self.config)
        self.client: Optional[OpenRouterClient] = None
        self.optimizer = ResponseOptimizer()
        self.features_used: List[str] = []
        
    async def initialize(self):
        """Inicializa el cliente de OpenRouter"""
        self.client = OpenRouterClient(self.config)
        await self.client.__aenter__()
        
    async def shutdown(self):
        """Cierra recursos"""
        if self.client:
            await self.client.__aexit__(None, None, None)
            
    async def chat(
        self,
        message: str,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Procesa un mensaje de chat con contexto y memoria
        """
        self.features_used = []
        
        # Guardar mensaje del usuario
        self.memory.add_memory(message, role="user", importance=1.0)
        
        # Construir contexto
        context = self.memory.get_context()
        
        # Comprimir según plan
        context = ContextCompressor.compress_for_plan(
            context, 
            self.config.plan
        )
        
        # Añadir system prompt según plan
        system_prompt = self._get_system_prompt()
        full_context = [{"role": "system", "content": system_prompt}] + context
        
        # Seleccionar modelo
        model = self._select_model()
        
        # Generar respuesta
        response_buffer = []
        
        async for chunk in self.client.chat_completion(
            messages=full_context,
            model=model,
            stream=stream
        ):
            response_buffer.append(chunk)
            yield chunk
            
        # Guardar respuesta completa
        full_response = "".join(response_buffer)
        
        # Optimizar según plan
        if self.config.plan == PlanType.FREE:
            full_response = self.optimizer.optimize_for_plan(
                full_response, 
                PlanType.FREE
            )
            
        # Guardar en memoria
        self.memory.add_memory(
            full_response, 
            role="assistant",
            importance=0.8
        )
        
    def _get_system_prompt(self) -> str:
        """System prompt adaptado al plan"""
        if self.config.plan == PlanType.FREE:
            return """Eres Apolo, un asistente AI amigable y eficiente.
Proporciona respuestas SENCILLAS y BREVES.
Usa lenguaje claro y directo.
Máximo 3 oraciones por respuesta."""
        else:
            return """Eres Apolo Plus, un asistente AI avanzado.
Proporciona respuestas DETALLADAS y PROFUNDAS.
Puedes realizar investigaciones complejas, análisis de código, y tareas automatizadas.
Usa todo el contexto disponible y sé exhaustivo en tus explicaciones.
Prioriza la precisión y la completitud."""
            
    def _select_model(self) -> Optional[str]:
        """Selecciona el mejor modelo disponible"""
        if self.config.plan == PlanType.PRO:
            # Pro usa modelos avanzados
            return self.config.models.get("chat_advanced") or self.config.models.get("chat")
        return self.config.models.get("chat")
        
    async def generate_image(self, prompt: str) -> Optional[str]:
        """Genera imágenes si está disponible"""
        if not self.config.can_use_feature("allow_image_generation"):
            return None
            
        self.features_used.append("image_generation")
        return await self.client.generate_image(prompt)
        
    async def web_search(self, query: str) -> List[Dict]:
        """Búsqueda web (solo Pro)"""
        if not self.config.can_use_feature("allow_web_search"):
            return []
            
        self.features_used.append("web_search")
        results = await self.client.web_search(query)
        
        # Guardar resultados en memoria
        for result in results:
            self.memory.add_memory(
                f"Web: {result.get('title')} - {result.get('snippet')}",
                role="system",
                importance=0.9,
                metadata={"source": "web_search", "url": result.get("url")}
            )
            
        return results
        
    async def analyze_code(self, code: str, language: str = "python"):
        """Análisis de código (solo Pro)"""
        if not self.config.can_use_feature("codex_access"):
            yield "🔒 Disponible en Apolo Plus"
            return
            
        self.features_used.append("codex_analysis")
        async for chunk in self.client.code_analysis(code, language):
            yield chunk
            
    def get_status(self) -> Dict:
        """Estado actual del sistema"""
        return {
            "plan": self.config.features.name,
            "memories_used": len(self.memory.memories),
            "memory_limit": self.config.features.memory_limit,
            "context_tokens": self.config.features.max_context_tokens,
            "features_available": [
                k for k, v in vars(self.config.features).items() 
                if isinstance(v, bool) and v
            ]
        }

class ApoloCLI:
    """Interfaz de línea de comandos"""
    
    def __init__(self):
        self.bot: Optional[ApoloXia] = None
        
    async def run(self):
        """Loop principal"""
        print("🚀 ApoloXia - Bot de IA")
        print("=" * 50)
        
        # Seleccionar plan
        plan_choice = input("Selecciona plan (1: Free, 2: Pro): ").strip()
        plan = PlanType.PRO if plan_choice == "2" else PlanType.FREE
        
        self.bot = ApoloXia(plan)
        await self.bot.initialize()
        
        print(f"\n✅ Plan activo: {self.bot.config.features.name}")
        print(f"💾 Memoria: {self.bot.config.features.memory_limit} mensajes")
        print(f"📝 Contexto: {self.bot.config.features.max_context_tokens} tokens")
        print("\nComandos especiales:")
        print("  /imagen <prompt> - Generar imagen")
        print("  /buscar <query>  - Búsqueda web (Pro)")
        print("  /codigo <archivo>- Analizar código (Pro)")
        print("  /estado          - Ver estado")
        print("  /salir           - Terminar")
        print("-" * 50)
        
        try:
            while True:
                user_input = input("\n👤 Tú: ").strip()
                
                if user_input.lower() == "/salir":
                    break
                    
                if user_input.lower() == "/estado":
                    status = self.bot.get_status()
                    print(f"\n📊 Estado: {status}")
                    continue
                    
                if user_input.lower().startswith("/imagen "):
                    prompt = user_input[8:]
                    print("🎨 Generando imagen...")
                    url = await self.bot.generate_image(prompt)
                    if url:
                        print(f"✅ Imagen: {url}")
                    else:
                        print("❌ No disponible en tu plan")
                    continue
                    
                if user_input.lower().startswith("/buscar "):
                    if self.bot.config.plan == PlanType.FREE:
                        print("🔒 Búsqueda web solo en Apolo Plus")
                        continue
                    query = user_input[8:]
                    print("🔍 Buscando...")
                    results = await self.bot.web_search(query)
                    for r in results[:3]:
                        print(f"  📄 {r.get('title')}: {r.get('snippet', '')[:100]}...")
                    continue
                
                # Chat normal
                print("🤖 Apolo: ", end="", flush=True)
                async for chunk in self.bot.chat(user_input):
                    print(chunk, end="", flush=True)
                print()
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
        finally:
            await self.bot.shutdown()

def main():
    """Entry point"""
    parser = argparse.ArgumentParser(description="ApoloXia - Bot de IA")
    parser.add_argument("--plan", choices=["free", "pro"], default="free",
                       help="Plan a usar")
    parser.add_argument("--demo", action="store_true",
                       help="Modo demo con ejemplos")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(run_demo(args.plan))
    else:
        cli = ApoloCLI()
        asyncio.run(cli.run())

async def run_demo(plan_str: str):
    """Demo rápido del sistema"""
    plan = PlanType.PRO if plan_str == "pro" else PlanType.FREE
    
    print(f"\n🎯 DEMO ApoloXia - Plan {plan.value.upper()}")
    print("=" * 60)
    
    bot = ApoloXia(plan)
    await bot.initialize()
    
    # Mostrar capacidades
    status = bot.get_status()
    print(f"\n📋 Capacidades activas:")
    for feature in status['features_available']:
        print(f"  ✅ {feature}")
    
    # Prueba de chat
    test_messages = [
        "Hola, ¿cómo estás?",
        "Explícame qué es la inteligencia artificial",
        "¿Puedes ayudarme con Python?"
    ]
    
    print(f"\n💬 Pruebas de conversación:")
    for msg in test_messages:
        print(f"\n👤 Usuario: {msg}")
        print("🤖 Apolo: ", end="", flush=True)
        
        response = []
        async for chunk in bot.chat(msg, stream=True):
            print(chunk, end="", flush=True)
            response.append(chunk)
        print()
        
        # En Free, mostrar que se trunca
        if plan == PlanType.FREE and len("".join(response)) > 100:
            print("   ⚡ [Respuesta optimizada para velocidad - Plan Free]")
    
    await bot.shutdown()
    print("\n✅ Demo completado")

if __name__ == "__main__":
    main()