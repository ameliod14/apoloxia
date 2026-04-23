from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

# Permitir CORS para cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir archivos estáticos (HTML, CSS, JS) desde el directorio actual
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Modelo para las peticiones de chat
class ChatRequest(BaseModel):
    user_id: str
    conversation_id: Optional[str] = None
    message: str
    tier: str = "free"
    agent_type: str = "general"
    model_id: Optional[str] = None
    use_web_search: bool = False
    enable_multi_agent: bool = False

# Endpoint de prueba
@app.get("/health")
async def health():
    return {"status": "ok", "mensaje": "Servidor funcionando correctamente"}

# Endpoint para obtener información del tier
@app.get("/tier-info/{user_id}")
async def tier_info(user_id: str):
    return {
        "tier": "free",
        "name": "Plan Gratuito",
        "remaining_daily": 100,
        "available_models": [
            {"id": "modelo1", "name": "Modelo Básico", "params": "1B", "context": "8K", "speed": "rápido"}
        ],
        "available_agents": ["general"],
        "features": {
            "web_search": False,
            "file_upload": False,
            "multi_agent": False
        }
    }

# Endpoint para listar conversaciones (vacío por ahora)
@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str):
    return {"user_id": user_id, "conversations": []}

# Endpoint para eliminar conversación
@app.delete("/conversations/{user_id}/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    return {"message": "Conversación eliminada (simulado)"}

# Endpoint principal de chat (simulado, solo eco)
@app.post("/chat")
async def chat(request: ChatRequest):
    return {
        "response": f"Eco del servidor: {request.message}",
        "agent_used": request.agent_type,
        "model_used": request.model_id or "modelo1",
        "remaining_daily": 99,
        "web_search_used": False
    }

# Redirigir /chat a chat.html para evitar error 405
@app.get("/chat")
async def redirect_chat():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/chat.html")

if __name__ == "__main__":
    print("🚀 Servidor NUEVO iniciado en http://localhost:8000")
    print("📌 Endpoints disponibles: /health, /tier-info/{user_id}, /conversations/{user_id}, /chat (POST), /chat (GET redirige a chat.html)")
    uvicorn.run(app, host="0.0.0.0", port=8000)