from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from agent import agent
from cache_memory import check_cache, update_cache  # 👈 Import cache functions
import os

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend (HTML, CSS, JS)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
async def get_homepage():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# Pydantic request model
class MessageRequest(BaseModel):
    message: str

# Chat endpoint with cache + agent fallback
@app.post("/ask")
async def ask(request: MessageRequest):
    try:
        query = request.message
        cached_reply = check_cache(query)

        if cached_reply:
            return {"response": cached_reply, "source": "cache"}  # 👈 reply from cache

        # Fallback to agent
        reply = agent.run(query)
        update_cache(query, reply)  # 👈 store in cache for future
        return {"response": reply, "source": "agent"}

    except Exception as e:
        return {"response": f"Agent error: {str(e)}", "source": "error"}
