from fastapi import FastAPI
from router import router

app = FastAPI(title="AssemblyAI Test API", version="0.1.0")

# Mount the routes
app.include_router(router)

# Health check
@app.get("/health")
async def health():
    return {"ok": True}
