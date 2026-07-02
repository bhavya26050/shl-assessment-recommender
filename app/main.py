import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.catalog import load_catalog
from app.retriever import CatalogRetriever
from app.agent import SHLAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global references
agent: SHLAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Loading SHL catalog...")
    catalog_path = os.path.join(os.path.dirname(__file__), "..", "data", "shl_product_catalog.json")
    catalog = load_catalog(catalog_path)
    logger.info(f"Loaded {len(catalog)} assessments")
    
    logger.info("Building search index...")
    retriever = CatalogRetriever(catalog)
    logger.info("Search index built")
    
    agent = SHLAgent(retriever)
    logger.info("Agent initialized and ready")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty")
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Service is still initializing")
    
    try:
        response = agent.process(request.messages)
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        return ChatResponse(
            reply="I apologize, but I encountered an error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
