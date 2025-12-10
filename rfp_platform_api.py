#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    RFP INTELLIGENCE PLATFORM API                               ║
║                     Built on Atomic Trust Kernel                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  Endpoints:                                                                    ║
║    GET  /                          - Health check & API info                   ║
║    GET  /api/rfps                  - List all RFPs                             ║
║    GET  /api/rfps/{id}             - Get RFP detail                            ║
║    POST /api/rfps                  - Create/ingest new RFP                     ║
║    POST /api/rfps/{id}/analyze     - Trigger analysis                          ║
║    POST /api/rfps/{id}/outcome     - Record actual outcome                     ║
║    GET  /api/capabilities          - List capabilities                         ║
║    GET  /api/stats                 - System statistics                         ║
║    GET  /api/analytics             - Full analytics data                       ║
║    WS   /ws/ingestion              - Real-time ingestion feed                  ║
║                                                                                ║
║  © 2024 Δtomic Artificial Intelligence Laboratory                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

# Optional: sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_EMBEDDER = True
except ImportError:
    HAS_EMBEDDER = False
    EMBEDDER = None

# Optional: BLAKE3 for hashing
try:
    import blake3
    def compute_hash(data: Dict[str, Any]) -> str:
        normalized = json.dumps(data, sort_keys=True, default=str)
        return blake3.blake3(normalized.encode()).hexdigest()[:24]
except ImportError:
    import hashlib
    def compute_hash(data: Dict[str, Any]) -> str:
        normalized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(normalized.encode()).hexdigest()[:24]

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s │ %(levelname)s │ %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("rfp-platform")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://atomic:AtomicTrust2025@localhost:5432/atomic_trust"
)
TRIBUNAL_URL = os.getenv("TRIBUNAL_URL", "http://localhost:3010")

# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CapabilityCreate(BaseModel):
    id: str
    name: str
    category: str
    maturity: str = "Developing"
    description: str
    evidence: List[str] = []
    sla: Optional[str] = None
    capacity: Optional[str] = None
    keywords: List[str] = []


class RequirementCreate(BaseModel):
    requirement_id: str
    text: str
    type: str  # MUST, SHALL, SHOULD
    category: str
    capability_match: Optional[str] = None
    match_score: int = 0
    gap: Optional[str] = None


class RFPCreate(BaseModel):
    name: str
    client: str
    industry: str
    value: Optional[str] = None
    deadline: Optional[str] = None
    category: Optional[str] = None
    difficulty: str = "Medium"
    requirements: List[RequirementCreate] = []


class VoteCreate(BaseModel):
    model: str
    verdict: str
    confidence: float
    rationale: str


class ConsensusCreate(BaseModel):
    verdict: str
    confidence: float
    phi: float
    psi: float
    delta: float
    phase: str
    clarifications: List[str] = []


class OutcomeCreate(BaseModel):
    outcome: str  # WIN, LOSS, CORRECT_NO_BID, VALIDATED_NO_BID
    notes: Optional[str] = None
    competitor_intel: Optional[str] = None


class AnalyzeRequest(BaseModel):
    votes: List[VoteCreate]
    consensus: ConsensusCreate


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═══════════════════════════════════════════════════════════════════════════════

db_pool: Optional[asyncpg.Pool] = None


async def get_db() -> asyncpg.Pool:
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
    return db_pool


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="RFP Intelligence Platform",
    description="AI-powered RFP analysis built on Atomic Trust Kernel",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connections for live updates
active_connections: List[WebSocket] = []


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP / SHUTDOWN
# ═══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("✓ Database connected")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_pool = None
    
    logger.info("RFP Intelligence Platform ready")


@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def row_to_dict(row: asyncpg.Record) -> Dict[str, Any]:
    """Convert asyncpg Record to dict with JSON serialization fixes."""
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, Decimal):
            d[k] = float(v)
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


async def broadcast(message: Dict[str, Any]):
    """Broadcast message to all WebSocket connections."""
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except:
            pass


def embed_text(text: str) -> Optional[List[float]]:
    """Generate embedding for text."""
    if HAS_EMBEDDER and EMBEDDER:
        return EMBEDDER.encode(text).tolist()
    return None


async def match_capability(pool: asyncpg.Pool, requirement_text: str, keywords: List[str] = None) -> Dict[str, Any]:
    """Find best matching capability for a requirement."""
    if not pool:
        return {"capability_id": None, "score": 0, "gap": None}
    
    # Try keyword matching first
    if keywords:
        async with pool.acquire() as conn:
            for kw in keywords:
                row = await conn.fetchrow("""
                    SELECT id, name, maturity FROM capabilities 
                    WHERE keywords @> $1::jsonb
                    LIMIT 1
                """, json.dumps([kw.lower()]))
                if row:
                    return {
                        "capability_id": row['id'],
                        "capability_name": row['name'],
                        "maturity": row['maturity'],
                        "score": 85 if row['maturity'] == 'Advanced' else 60,
                        "gap": None if row['maturity'] == 'Advanced' else "Capability not fully mature"
                    }
    
    # Fallback to embedding similarity if available
    if HAS_EMBEDDER and EMBEDDER:
        embedding = embed_text(requirement_text)
        if embedding:
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT id, name, maturity, 
                           1 - (embedding <=> $1::vector) as similarity
                    FROM capabilities
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT 1
                """, embedding)
                if row and row['similarity'] > 0.5:
                    score = int(row['similarity'] * 100)
                    return {
                        "capability_id": row['id'],
                        "capability_name": row['name'],
                        "maturity": row['maturity'],
                        "score": score,
                        "gap": None if score >= 80 else "Partial capability match"
                    }
    
    return {"capability_id": None, "score": 0, "gap": "No matching capability found"}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    """Serve the RFP Platform demo UI."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Demo not found. Place index.html in the API directory.</h1>"

@app.get("/")
async def root():
    """Health check and API info."""
    return {
        "service": "RFP Intelligence Platform",
        "version": "1.0.0",
        "kernel": "Atomic Trust v4.0",
        "database": "connected" if db_pool else "disconnected",
        "embedder": "loaded" if HAS_EMBEDDER else "not available",
        "endpoints": {
            "GET /api/rfps": "List all RFPs",
            "GET /api/rfps/{id}": "Get RFP detail with requirements",
            "POST /api/rfps": "Create new RFP",
            "POST /api/rfps/{id}/analyze": "Record analysis results",
            "POST /api/rfps/{id}/outcome": "Record actual outcome",
            "GET /api/capabilities": "List all capabilities",
            "POST /api/capabilities": "Create capability",
            "GET /api/stats": "System statistics",
            "GET /api/analytics": "Full analytics data",
            "WS /ws/ingestion": "Real-time ingestion feed"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    db_ok = False
    if db_pool:
        try:
            async with db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            db_ok = True
        except:
            pass
    
    return {
        "status": "healthy" if db_ok else "degraded",
        "database": db_ok,
        "embedder": HAS_EMBEDDER,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ─────────────────────────────────────────────────────────────────────────────
# RFP ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/rfps")
async def list_rfps(
    status: Optional[str] = None,
    industry: Optional[str] = None,
    verdict: Optional[str] = None,
    limit: int = 50
):
    """List all RFPs with optional filters."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with db_pool.acquire() as conn:
        query = """
            SELECT r.*, 
                   (SELECT COUNT(*) FROM rfp_requirements WHERE rfp_id = r.id) as requirement_count,
                   (SELECT AVG(match_score) FROM rfp_requirements WHERE rfp_id = r.id) as avg_match_score
            FROM rfps r
            WHERE 1=1
        """
        params = []
        param_idx = 1
        
        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1
        
        if industry:
            query += f" AND industry = ${param_idx}"
            params.append(industry)
            param_idx += 1
        
        if verdict:
            query += f" AND verdict = ${param_idx}"
            params.append(verdict)
            param_idx += 1
        
        query += f" ORDER BY created_at DESC LIMIT ${param_idx}"
        params.append(limit)
        
        rows = await conn.fetch(query, *params)
        return [row_to_dict(r) for r in rows]


@app.get("/api/rfps/{rfp_id}")
async def get_rfp(rfp_id: str):
    """Get RFP detail with requirements, votes, and clarifications."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with db_pool.acquire() as conn:
        # Get RFP
        rfp = await conn.fetchrow("SELECT * FROM rfps WHERE id = $1", rfp_id)
        if not rfp:
            raise HTTPException(status_code=404, detail="RFP not found")
        
        result = row_to_dict(rfp)
        
        # Get requirements with capability info
        reqs = await conn.fetch("""
            SELECT r.*, c.name as capability_name, c.maturity as capability_maturity
            FROM rfp_requirements r
            LEFT JOIN capabilities c ON r.matched_capability_id = c.id
            WHERE r.rfp_id = $1
            ORDER BY r.requirement_id
        """, rfp_id)
        result['requirements'] = [row_to_dict(r) for r in reqs]
        
        # Get votes
        votes = await conn.fetch("""
            SELECT * FROM rfp_tribunal_votes
            WHERE rfp_id = $1
            ORDER BY created_at
        """, rfp_id)
        result['votes'] = [row_to_dict(v) for v in votes]
        
        # Get clarifications
        clarifications = await conn.fetch("""
            SELECT * FROM rfp_clarifications
            WHERE rfp_id = $1
            ORDER BY created_at
        """, rfp_id)
        result['clarifications'] = [row_to_dict(c) for c in clarifications]
        
        # Get similar RFPs
        similar = await conn.fetch("""
            SELECT s.similar_rfp_id, s.similarity_score, r.name, r.client, r.verdict, r.actual_outcome
            FROM rfp_similar s
            JOIN rfps r ON s.similar_rfp_id = r.id
            WHERE s.rfp_id = $1
            ORDER BY s.similarity_score DESC
            LIMIT 5
        """, rfp_id)
        result['similar_rfps'] = [row_to_dict(s) for s in similar]
        
        return result


@app.post("/api/rfps")
async def create_rfp(rfp: RFPCreate):
    """Create a new RFP with requirements."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    rfp_id = str(uuid4())[:8]
    
    async with db_pool.acquire() as conn:
        async with conn.transaction():
            # Create RFP
            await conn.execute("""
                INSERT INTO rfps (id, name, client, industry, value, deadline, category, difficulty, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'pending')
            """, rfp_id, rfp.name, rfp.client, rfp.industry, rfp.value, rfp.deadline, rfp.category, rfp.difficulty)
            
            # Create requirements
            for req in rfp.requirements:
                req_id = str(uuid4())
                
                # Match capability
                match = await match_capability(db_pool, req.text)
                
                await conn.execute("""
                    INSERT INTO rfp_requirements 
                    (id, rfp_id, requirement_id, text, type, category, matched_capability_id, match_score, gap_description)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, req_id, rfp_id, req.requirement_id, req.text, req.type, req.category,
                   req.capability_match or match.get('capability_id'),
                   req.match_score or match.get('score', 0),
                   req.gap or match.get('gap'))
            
            logger.info(f"Created RFP {rfp_id}: {rfp.name}")
    
    # Broadcast creation
    await broadcast({
        "type": "rfp_created",
        "rfp_id": rfp_id,
        "name": rfp.name,
        "client": rfp.client
    })
    
    return {"id": rfp_id, "status": "created"}


@app.post("/api/rfps/{rfp_id}/analyze")
async def analyze_rfp(rfp_id: str, analysis: AnalyzeRequest):
    """Record analysis results (votes and consensus) for an RFP."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with db_pool.acquire() as conn:
        # Verify RFP exists
        rfp = await conn.fetchrow("SELECT * FROM rfps WHERE id = $1", rfp_id)
        if not rfp:
            raise HTTPException(status_code=404, detail="RFP not found")
        
        async with conn.transaction():
            # Get chain position for votes
            chain_seq = await conn.fetchval(
                "SELECT COALESCE(MAX(chain_sequence), 0) + 1 FROM rfp_tribunal_votes"
            )
            prev_hash = await conn.fetchval(
                "SELECT record_hash FROM rfp_tribunal_votes ORDER BY chain_sequence DESC LIMIT 1"
            ) or "GENESIS"
            
            # Record votes
            for vote in analysis.votes:
                vote_hash = compute_hash({
                    "rfp_id": rfp_id,
                    "model": vote.model,
                    "verdict": vote.verdict,
                    "confidence": vote.confidence,
                    "chain_sequence": chain_seq
                })
                
                await conn.execute("""
                    INSERT INTO rfp_tribunal_votes 
                    (rfp_id, model_id, verdict, confidence, rationale, record_hash, previous_hash, chain_sequence)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, rfp_id, vote.model, vote.verdict, vote.confidence, vote.rationale,
                   vote_hash, prev_hash, chain_seq)
                
                prev_hash = vote_hash
                chain_seq += 1
            
            # Calculate average capability match
            avg_match = await conn.fetchval(
                "SELECT AVG(match_score) FROM rfp_requirements WHERE rfp_id = $1", rfp_id
            ) or 0
            
            # Update RFP with consensus
            await conn.execute("""
                UPDATE rfps SET
                    status = 'complete',
                    verdict = $2,
                    confidence = $3,
                    phi = $4,
                    psi = $5,
                    delta = $6,
                    phase = $7,
                    capability_match_avg = $8,
                    analyzed_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1
            """, rfp_id, analysis.consensus.verdict, analysis.consensus.confidence,
               analysis.consensus.phi, analysis.consensus.psi, analysis.consensus.delta,
               analysis.consensus.phase, avg_match)
            
            # Record clarifications
            for clarification in analysis.consensus.clarifications:
                await conn.execute("""
                    INSERT INTO rfp_clarifications (rfp_id, text, status)
                    VALUES ($1, $2, 'open')
                """, rfp_id, clarification)
    
    # Broadcast analysis complete
    await broadcast({
        "type": "analysis_complete",
        "rfp_id": rfp_id,
        "verdict": analysis.consensus.verdict,
        "confidence": analysis.consensus.confidence
    })
    
    return {
        "rfp_id": rfp_id,
        "status": "analyzed",
        "verdict": analysis.consensus.verdict,
        "confidence": analysis.consensus.confidence
    }


@app.post("/api/rfps/{rfp_id}/outcome")
async def record_outcome(rfp_id: str, outcome: OutcomeCreate):
    """Record actual outcome for an RFP and update credentials."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with db_pool.acquire() as conn:
        # Verify RFP exists and is analyzed
        rfp = await conn.fetchrow("SELECT * FROM rfps WHERE id = $1", rfp_id)
        if not rfp:
            raise HTTPException(status_code=404, detail="RFP not found")
        if rfp['status'] != 'complete':
            raise HTTPException(status_code=400, detail="RFP not yet analyzed")
        
        # Update RFP with outcome
        await conn.execute("""
            UPDATE rfps SET
                actual_outcome = $2,
                outcome_notes = $3,
                competitor_intel = $4,
                outcome_at = NOW(),
                updated_at = NOW()
            WHERE id = $1
        """, rfp_id, outcome.outcome, outcome.notes, outcome.competitor_intel)
        
        # Update system stats
        await conn.execute("SELECT update_system_stats()")
        await conn.execute("SELECT update_industry_performance()")
    
    # Broadcast outcome
    await broadcast({
        "type": "outcome_recorded",
        "rfp_id": rfp_id,
        "outcome": outcome.outcome
    })
    
    return {
        "rfp_id": rfp_id,
        "outcome": outcome.outcome,
        "status": "recorded"
    }


# ─────────────────────────────────────────────────────────────────────────────
# CAPABILITY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/capabilities")
async def list_capabilities(category: Optional[str] = None):
    """List all capabilities."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with db_pool.acquire() as conn:
        if category:
            rows = await conn.fetch(
                "SELECT * FROM capabilities WHERE category = $1 ORDER BY name", 
                category
            )
        else:
            rows = await conn.fetch("SELECT * FROM capabilities ORDER BY category, name")
        
        return [row_to_dict(r) for r in rows]


@app.post("/api/capabilities")
async def create_capability(cap: CapabilityCreate):
    """Create a new capability."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    embedding = embed_text(f"{cap.name} {cap.description} {' '.join(cap.keywords)}")
    
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO capabilities (id, name, category, maturity, description, evidence, sla, capacity, keywords, embedding)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                maturity = EXCLUDED.maturity,
                description = EXCLUDED.description,
                evidence = EXCLUDED.evidence,
                sla = EXCLUDED.sla,
                capacity = EXCLUDED.capacity,
                keywords = EXCLUDED.keywords,
                embedding = EXCLUDED.embedding,
                updated_at = NOW()
        """, cap.id, cap.name, cap.category, cap.maturity, cap.description,
           json.dumps(cap.evidence), cap.sla, cap.capacity, json.dumps(cap.keywords),
           embedding)
    
    return {"id": cap.id, "status": "created"}


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS & ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    """Get system statistics."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with db_pool.acquire() as conn:
        stats = await conn.fetchrow("SELECT * FROM system_stats WHERE id = 1")
        if stats:
            return row_to_dict(stats)
        return {
            "total_decisions": 0,
            "accuracy": 0,
            "bid_win_rate": 0,
            "no_bid_accuracy": 0,
            "avg_confidence": 0
        }


@app.get("/api/analytics")
async def get_analytics():
    """Get full analytics data for dashboard."""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")
    
    async with db_pool.acquire() as conn:
        # System stats
        stats = await conn.fetchrow("SELECT * FROM system_stats WHERE id = 1")
        
        # Accuracy history
        accuracy_history = await conn.fetch(
            "SELECT month, accuracy FROM accuracy_history ORDER BY month LIMIT 12"
        )
        
        # Industry performance
        industry_perf = await conn.fetch("SELECT * FROM industry_performance")
        
        # Model performance
        model_perf = await conn.fetch("SELECT * FROM model_performance")
        
        # Recent RFPs
        recent = await conn.fetch("""
            SELECT id, name, client, industry, value, verdict, confidence, actual_outcome
            FROM rfps
            ORDER BY created_at DESC
            LIMIT 10
        """)
        
        return {
            "stats": row_to_dict(stats) if stats else {},
            "accuracy_history": [row_to_dict(r) for r in accuracy_history],
            "by_industry": {r['industry']: row_to_dict(r) for r in industry_perf},
            "model_performance": [row_to_dict(r) for r in model_perf],
            "recent_rfps": [row_to_dict(r) for r in recent]
        }


@app.get("/api/credentials")
async def get_credentials():
    """Get model credentials (proxies to tribunal API or returns from DB)."""
    # Try to get from tribunal API
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{TRIBUNAL_URL}/tribunal/credentials", timeout=5)
            if resp.status_code == 200:
                return resp.json()
    except:
        pass
    
    # Fallback to local DB
    if db_pool:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id as model_id, omega, psi, total_votes, correct_votes,
                       CASE WHEN total_votes > 0 THEN correct_votes::float / total_votes ELSE 0 END as accuracy
                FROM llm_agents
            """)
            return {"models": [row_to_dict(r) for r in rows]}
    
    return {"models": []}


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET FOR LIVE UPDATES
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/ingestion")
async def websocket_ingestion(websocket: WebSocket):
    """WebSocket for real-time ingestion updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive, receive any client messages
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_json({"type": "pong", "data": data})
    except WebSocketDisconnect:
        active_connections.remove(websocket)


# ─────────────────────────────────────────────────────────────────────────────
# FRONTEND SERVING
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/app", response_class=HTMLResponse)
async def serve_app():
    """Serve the frontend application."""
    # Look for the HTML file
    html_paths = [
        "/mnt/user-data/outputs/atomic_trust_rfp_platform.html",
        "./atomic_trust_rfp_platform.html",
        "../atomic_trust_rfp_platform.html"
    ]
    
    for path in html_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return HTMLResponse(content=f.read())
    
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3020)
