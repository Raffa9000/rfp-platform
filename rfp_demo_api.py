#!/usr/bin/env python3
"""
RFP INTELLIGENCE PLATFORM - LIVE DEMO API
==========================================
Real-time RFP analysis with Multi-LLM Tribunal

Features:
- 5 Sample RFPs ready for analysis
- Real-time processing visualization
- Multi-LLM voting (Claude, GPT-4, Gemini)
- Streaming progress updates via SSE
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, Response
from pydantic import BaseModel

# LLM Clients
import httpx

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# SIMULATION MODE - Set to True for demo without real API calls
SIMULATION_MODE = True

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Mock LLM responses for simulation mode
MOCK_RESPONSES = {
    "claude": {
        "verdict": "BID",
        "confidence": 0.87,
        "rationale": "Strong technical alignment with our cloud migration expertise. The healthcare compliance requirements (HIPAA) match our proven track record. Key strengths: microservices architecture experience, established Azure partnership, and dedicated DevSecOps team. Risk factors are manageable with proper resource allocation. Recommend pursuing with emphasis on security credentials."
    },
    "gpt4": {
        "verdict": "STRONG_BID",
        "confidence": 0.92,
        "rationale": "Excellent opportunity alignment. Our capabilities exceed 85% of stated requirements. The budget is well within our typical engagement range, and the timeline is achievable with our standard methodology. Competitive advantage: our proprietary migration framework reduces deployment risk by 40%. Strong recommendation to bid with premium pricing tier."
    },
    "grok": {
        "verdict": "BID",
        "confidence": 0.78,
        "rationale": "Solid fit with moderate complexity. Technical requirements align well, though legacy system integration presents some challenges. Our team has relevant experience but may need to augment with specialized consultants. The compliance matrix is demanding but achievable. Suggest bidding with contingency provisions."
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLE RFPs
# ═══════════════════════════════════════════════════════════════════════════════

SAMPLE_RFPS = {
    "rfp-001": {
        "id": "rfp-001",
        "name": "Cloud Migration & Modernization",
        "client": "Pacific Northwest Healthcare",
        "industry": "Healthcare",
        "value": "$2.4M",
        "deadline": "45 days",
        "difficulty": "High",
        "summary": "Enterprise cloud migration for 12 hospital network with HIPAA compliance requirements.",
        "requirements": [
            {"id": "REQ-001", "type": "MUST", "category": "Compliance", "text": "All systems must maintain HIPAA compliance throughout migration and post-deployment"},
            {"id": "REQ-002", "type": "MUST", "category": "Technical", "text": "Zero-downtime migration for critical patient care systems (EMR, PACS, Lab)"},
            {"id": "REQ-003", "type": "SHALL", "category": "Security", "text": "Implement end-to-end encryption for all data in transit and at rest"},
            {"id": "REQ-004", "type": "SHOULD", "category": "Performance", "text": "Achieve 99.99% uptime SLA for Tier-1 applications"},
            {"id": "REQ-005", "type": "MUST", "category": "Timeline", "text": "Complete migration within 18-month window with quarterly milestones"},
            {"id": "REQ-006", "type": "SHALL", "category": "Training", "text": "Provide comprehensive training for 2,500+ clinical and IT staff"},
        ],
        "full_text": """
REQUEST FOR PROPOSAL: Enterprise Cloud Migration & Modernization Services

Pacific Northwest Healthcare (PNWH) is seeking qualified vendors to provide comprehensive
cloud migration and modernization services for our 12-hospital network serving over 2 million
patients annually.

SCOPE OF WORK:
The selected vendor will migrate our legacy on-premises infrastructure to a cloud-native
architecture while maintaining full HIPAA compliance and zero disruption to patient care.

KEY REQUIREMENTS:
1. HIPAA Compliance - All systems must maintain full HIPAA compliance
2. Zero-Downtime Migration - Critical systems cannot experience any downtime
3. Data Security - End-to-end encryption required for all patient data
4. High Availability - 99.99% uptime SLA for critical applications
5. Timeline - 18-month implementation with quarterly milestones
6. Training - Comprehensive training for 2,500+ staff members

EVALUATION CRITERIA:
- Technical Approach (35%)
- Past Performance in Healthcare (25%)
- Price/Cost (20%)
- Team Qualifications (20%)

BUDGET: $2.4M - $3.2M
DEADLINE: Proposals due in 45 days
        """
    },
    "rfp-002": {
        "id": "rfp-002",
        "name": "AI-Powered Fraud Detection System",
        "client": "First National Banking Corp",
        "industry": "Financial Services",
        "value": "$1.8M",
        "deadline": "30 days",
        "difficulty": "High",
        "summary": "Real-time fraud detection platform using machine learning for transaction monitoring.",
        "requirements": [
            {"id": "REQ-001", "type": "MUST", "category": "Performance", "text": "Process 50,000+ transactions per second with sub-100ms latency"},
            {"id": "REQ-002", "type": "MUST", "category": "Accuracy", "text": "Achieve 99.5% fraud detection rate with less than 0.1% false positive rate"},
            {"id": "REQ-003", "type": "SHALL", "category": "Compliance", "text": "Full PCI-DSS Level 1 compliance and SOC 2 Type II certification"},
            {"id": "REQ-004", "type": "MUST", "category": "Integration", "text": "Seamless integration with existing core banking platform (FIS)"},
            {"id": "REQ-005", "type": "SHOULD", "category": "ML", "text": "Self-learning models that adapt to emerging fraud patterns"},
        ],
        "full_text": """
REQUEST FOR PROPOSAL: AI-Powered Fraud Detection System

First National Banking Corp seeks an advanced fraud detection solution leveraging
artificial intelligence and machine learning to protect our 8 million customers.

BACKGROUND:
Current fraud losses exceed $45M annually. We require a next-generation solution
capable of real-time detection across all transaction channels.

TECHNICAL REQUIREMENTS:
- High throughput: 50,000+ TPS with sub-100ms response
- High accuracy: 99.5% detection, <0.1% false positives
- PCI-DSS Level 1 compliant
- Integration with FIS core banking platform
- Adaptive ML models for emerging threats

BUDGET: $1.8M implementation + ongoing licensing
TIMELINE: 30 days for proposal, 9-month implementation
        """
    },
    "rfp-003": {
        "id": "rfp-003",
        "name": "Smart City IoT Platform",
        "client": "Metro Denver Municipal Authority",
        "industry": "Government",
        "value": "$4.2M",
        "deadline": "60 days",
        "difficulty": "Very High",
        "summary": "City-wide IoT infrastructure for traffic, utilities, and public safety integration.",
        "requirements": [
            {"id": "REQ-001", "type": "MUST", "category": "Scale", "text": "Support 500,000+ IoT devices across 150 square mile coverage area"},
            {"id": "REQ-002", "type": "MUST", "category": "Security", "text": "FedRAMP Moderate authorization required for all cloud components"},
            {"id": "REQ-003", "type": "SHALL", "category": "Integration", "text": "Integrate with existing 911 dispatch, traffic management, and utility SCADA systems"},
            {"id": "REQ-004", "type": "MUST", "category": "Resilience", "text": "Operate independently during network outages with 72-hour edge autonomy"},
            {"id": "REQ-005", "type": "SHOULD", "category": "Analytics", "text": "Predictive analytics for traffic optimization and infrastructure maintenance"},
            {"id": "REQ-006", "type": "MUST", "category": "Procurement", "text": "Comply with Buy American Act and local procurement preferences"},
        ],
        "full_text": """
REQUEST FOR PROPOSAL: Smart City IoT Platform Implementation

Metro Denver Municipal Authority invites proposals for a comprehensive Smart City
IoT platform to modernize city infrastructure and improve citizen services.

PROJECT SCOPE:
- Traffic management system with 2,000+ smart intersections
- Utility monitoring (water, electric, gas) with 400,000+ meters
- Public safety integration with gunshot detection and emergency response
- Environmental monitoring (air quality, noise, weather)

KEY REQUIREMENTS:
- Scale to 500,000+ IoT devices
- FedRAMP Moderate authorization
- Integration with legacy systems
- 72-hour edge autonomy during outages
- Predictive analytics capabilities
- Buy American Act compliance

BUDGET: $4.2M over 3 years
PROPOSAL DEADLINE: 60 days
        """
    },
    "rfp-004": {
        "id": "rfp-004",
        "name": "E-Commerce Platform Rebuild",
        "client": "Outdoor Gear Collective",
        "industry": "Retail",
        "value": "$650K",
        "deadline": "21 days",
        "difficulty": "Medium",
        "summary": "Modern headless commerce platform to replace legacy Magento installation.",
        "requirements": [
            {"id": "REQ-001", "type": "MUST", "category": "Performance", "text": "Sub-2 second page load times on mobile devices"},
            {"id": "REQ-002", "type": "SHALL", "category": "Integration", "text": "Integration with existing ERP (NetSuite) and WMS systems"},
            {"id": "REQ-003", "type": "SHOULD", "category": "Features", "text": "AI-powered product recommendations and search"},
            {"id": "REQ-004", "type": "MUST", "category": "Migration", "text": "Migrate 50,000 SKUs and 8 years of customer data without loss"},
            {"id": "REQ-005", "type": "SHOULD", "category": "Mobile", "text": "Progressive Web App with offline catalog browsing"},
        ],
        "full_text": """
REQUEST FOR PROPOSAL: E-Commerce Platform Modernization

Outdoor Gear Collective, a leading outdoor recreation retailer with $85M annual
online revenue, seeks to replace our aging Magento 1 platform.

OBJECTIVES:
- Improve site performance and conversion rates
- Enable omnichannel capabilities (BOPIS, ship-from-store)
- Modernize technology stack for easier maintenance

REQUIREMENTS:
- Sub-2 second mobile page loads
- NetSuite and WMS integration
- AI-powered recommendations
- Full data migration (50K SKUs, 8 years history)
- PWA capabilities

BUDGET: $650K
TIMELINE: 21-day proposal window, 6-month implementation
        """
    },
    "rfp-005": {
        "id": "rfp-005",
        "name": "Cybersecurity Operations Center",
        "client": "Meridian Energy Partners",
        "industry": "Energy",
        "value": "$3.1M",
        "deadline": "45 days",
        "difficulty": "High",
        "summary": "24/7 Security Operations Center with OT/IT convergence for critical infrastructure.",
        "requirements": [
            {"id": "REQ-001", "type": "MUST", "category": "Compliance", "text": "NERC CIP compliance for all bulk electric system components"},
            {"id": "REQ-002", "type": "MUST", "category": "Operations", "text": "24/7/365 monitoring with 15-minute incident response SLA"},
            {"id": "REQ-003", "type": "SHALL", "category": "Coverage", "text": "Monitor both IT infrastructure and OT/ICS/SCADA systems"},
            {"id": "REQ-004", "type": "MUST", "category": "Staffing", "text": "US-based SOC analysts with Secret clearance eligibility"},
            {"id": "REQ-005", "type": "SHOULD", "category": "Threat Intel", "text": "Integration with E-ISAC and sector-specific threat intelligence"},
            {"id": "REQ-006", "type": "MUST", "category": "Recovery", "text": "Documented incident response and disaster recovery procedures"},
        ],
        "full_text": """
REQUEST FOR PROPOSAL: Cybersecurity Operations Center Services

Meridian Energy Partners operates 4,200 MW of generation capacity and 3,500 miles
of transmission lines. We require a managed SOC to protect critical infrastructure.

SCOPE:
- 24/7 security monitoring and incident response
- IT and OT/SCADA system coverage
- NERC CIP compliance management
- Threat hunting and vulnerability management

REQUIREMENTS:
- NERC CIP compliance
- 24/7/365 with 15-min response SLA
- IT/OT convergence capabilities
- US-based, clearance-eligible staff
- E-ISAC threat intel integration
- Documented IR/DR procedures

BUDGET: $3.1M annually
PROPOSAL DUE: 45 days
        """
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY DATABASE (Company's capabilities)
# ═══════════════════════════════════════════════════════════════════════════════

CAPABILITIES = {
    "cloud-migration": {
        "name": "Cloud Migration Services",
        "maturity": "Advanced",
        "description": "Full-stack cloud migration including lift-and-shift, re-platforming, and cloud-native refactoring",
        "certifications": ["AWS Advanced Partner", "Azure Expert MSP", "GCP Premier Partner"],
        "past_projects": 47,
        "industries": ["Healthcare", "Financial Services", "Retail"]
    },
    "healthcare-compliance": {
        "name": "Healthcare IT & Compliance",
        "maturity": "Advanced",
        "description": "HIPAA-compliant healthcare solutions including EMR integration and clinical workflows",
        "certifications": ["HITRUST CSF", "SOC 2 Type II"],
        "past_projects": 23,
        "industries": ["Healthcare"]
    },
    "ai-ml": {
        "name": "AI/ML Solutions",
        "maturity": "Developing",
        "description": "Machine learning model development, MLOps, and AI integration",
        "certifications": ["AWS ML Specialty"],
        "past_projects": 12,
        "industries": ["Financial Services", "Retail"]
    },
    "fraud-detection": {
        "name": "Fraud Detection Systems",
        "maturity": "Basic",
        "description": "Transaction monitoring and anomaly detection",
        "certifications": [],
        "past_projects": 3,
        "industries": ["Financial Services"]
    },
    "iot-platforms": {
        "name": "IoT Platform Development",
        "maturity": "Developing",
        "description": "IoT architecture, edge computing, and device management",
        "certifications": ["AWS IoT Competency"],
        "past_projects": 8,
        "industries": ["Manufacturing", "Retail"]
    },
    "government": {
        "name": "Government & Public Sector",
        "maturity": "Basic",
        "description": "FedRAMP and government contract experience",
        "certifications": [],
        "past_projects": 2,
        "industries": ["Government"]
    },
    "ecommerce": {
        "name": "E-Commerce Solutions",
        "maturity": "Advanced",
        "description": "Headless commerce, Shopify Plus, and custom platforms",
        "certifications": ["Shopify Plus Partner", "BigCommerce Elite"],
        "past_projects": 34,
        "industries": ["Retail"]
    },
    "cybersecurity": {
        "name": "Cybersecurity Services",
        "maturity": "Developing",
        "description": "Security assessments, SOC services, and incident response",
        "certifications": ["ISO 27001"],
        "past_projects": 15,
        "industries": ["Financial Services", "Healthcare"]
    },
    "ot-security": {
        "name": "OT/ICS Security",
        "maturity": "Basic",
        "description": "Operational technology and industrial control system security",
        "certifications": [],
        "past_projects": 1,
        "industries": ["Energy"]
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# LLM INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def sanitize_error(error_msg: str) -> str:
    """Remove any API keys or sensitive data from error messages."""
    import re
    # Remove common API key patterns
    patterns = [
        r'sk-[a-zA-Z0-9_-]{20,}',  # OpenAI keys
        r'sk-ant-[a-zA-Z0-9_-]{20,}',  # Anthropic keys
        r'xai-[a-zA-Z0-9_-]{20,}',  # xAI keys
        r'Bearer [a-zA-Z0-9_-]{20,}',  # Bearer tokens
        r'x-api-key["\s:]+[a-zA-Z0-9_-]{20,}',  # x-api-key headers
    ]
    result = str(error_msg)
    for pattern in patterns:
        result = re.sub(pattern, '[REDACTED]', result, flags=re.IGNORECASE)
    return result

async def call_claude(prompt: str, system: str = None) -> Dict[str, Any]:
    """Call Claude API for analysis."""
    if not ANTHROPIC_API_KEY:
        return {"error": "ANTHROPIC_API_KEY not configured", "model": "claude-3-5-sonnet"}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            messages = [{"role": "user", "content": prompt}]
            payload = {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 2000,
                "messages": messages
            }
            if system:
                payload["system"] = system

            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload
            )

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "model": "claude-sonnet-4",
                    "response": data["content"][0]["text"],
                    "success": True
                }
            else:
                return {"error": f"API error: {resp.status_code}", "model": "claude-sonnet-4"}
        except Exception as e:
            return {"error": sanitize_error(str(e)), "model": "claude-sonnet-4"}


async def call_openai(prompt: str, system: str = None) -> Dict[str, Any]:
    """Call OpenAI GPT-4 for analysis."""
    if not OPENAI_API_KEY:
        return {"error": "OPENAI_API_KEY not configured", "model": "gpt-4o"}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o",
                    "messages": messages,
                    "max_tokens": 2000
                }
            )

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "model": "gpt-4o",
                    "response": data["choices"][0]["message"]["content"],
                    "success": True
                }
            else:
                return {"error": f"API error: {resp.status_code}", "model": "gpt-4o"}
        except Exception as e:
            return {"error": sanitize_error(str(e)), "model": "gpt-4o"}


async def call_grok(prompt: str, system: str = None) -> Dict[str, Any]:
    """Call xAI Grok for analysis."""
    if not XAI_API_KEY:
        return {"error": "XAI_API_KEY not configured", "model": "grok-2"}

    async with httpx.AsyncClient(timeout=60) as client:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {XAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-2-latest",
                    "messages": messages,
                    "max_tokens": 2000
                }
            )

            if resp.status_code == 200:
                data = resp.json()
                return {
                    "model": "grok-2",
                    "response": data["choices"][0]["message"]["content"],
                    "success": True
                }
            else:
                return {"error": f"API error: {resp.status_code}", "model": "grok-2"}
        except Exception as e:
            return {"error": sanitize_error(str(e)), "model": "grok-2"}


def parse_vote(response: str) -> Dict[str, Any]:
    """Parse LLM response into structured vote."""
    text = response.lower()

    # Determine verdict
    if "strong bid" in text or "definitely bid" in text:
        verdict = "STRONG_BID"
        confidence = 0.9
    elif "bid" in text and "no" not in text[:50]:
        verdict = "BID"
        confidence = 0.75
    elif "no bid" in text or "no-bid" in text or "pass" in text:
        verdict = "NO_BID"
        confidence = 0.8
    elif "conditional" in text or "maybe" in text:
        verdict = "CONDITIONAL"
        confidence = 0.6
    else:
        verdict = "REVIEW"
        confidence = 0.5

    # Extract rationale (first 500 chars of response)
    rationale = response[:500] if len(response) > 500 else response

    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": rationale
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="RFP Intelligence Platform - Live Demo", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API info and status."""
    return {
        "service": "RFP Intelligence Platform",
        "version": "2.0.0 - Live Demo",
        "llm_status": {
            "claude": "simulated" if SIMULATION_MODE else ("configured" if ANTHROPIC_API_KEY else "not configured"),
            "gpt4": "simulated" if SIMULATION_MODE else ("configured" if OPENAI_API_KEY else "not configured"),
            "grok": "simulated" if SIMULATION_MODE else ("configured" if XAI_API_KEY else "not configured")
        },
        "simulation_mode": SIMULATION_MODE,
        "sample_rfps": len(SAMPLE_RFPS),
        "capabilities": len(CAPABILITIES)
    }


@app.get("/api/sample-rfps")
async def list_sample_rfps():
    """List all sample RFPs for demo."""
    return [
        {
            "id": rfp["id"],
            "name": rfp["name"],
            "client": rfp["client"],
            "industry": rfp["industry"],
            "value": rfp["value"],
            "deadline": rfp["deadline"],
            "difficulty": rfp["difficulty"],
            "summary": rfp["summary"],
            "requirement_count": len(rfp["requirements"])
        }
        for rfp in SAMPLE_RFPS.values()
    ]


@app.get("/api/sample-rfps/{rfp_id}")
async def get_sample_rfp(rfp_id: str):
    """Get full details of a sample RFP."""
    if rfp_id not in SAMPLE_RFPS:
        raise HTTPException(status_code=404, detail="RFP not found")
    return SAMPLE_RFPS[rfp_id]


@app.get("/api/capabilities")
async def list_capabilities():
    """List company capabilities."""
    return CAPABILITIES


@app.get("/api/analyze/{rfp_id}")
async def analyze_rfp_stream(rfp_id: str):
    """Stream real-time RFP analysis with LLM tribunal voting."""
    if rfp_id not in SAMPLE_RFPS:
        raise HTTPException(status_code=404, detail="RFP not found")

    rfp = SAMPLE_RFPS[rfp_id]

    async def generate():
        rfp_name = rfp["name"]
        req_count = len(rfp["requirements"])

        # Phase 1: Document Processing
        yield f"data: {json.dumps({'phase': 'processing', 'step': 'start', 'message': 'Starting RFP Analysis Pipeline...'})}\n\n"
        await asyncio.sleep(0.5)

        yield f"data: {json.dumps({'phase': 'processing', 'step': 'load', 'message': 'Loading RFP: ' + rfp_name})}\n\n"
        await asyncio.sleep(0.3)

        yield f"data: {json.dumps({'phase': 'processing', 'step': 'parse', 'message': 'Parsing ' + str(req_count) + ' requirements...'})}\n\n"
        await asyncio.sleep(0.5)

        # Phase 2: Requirement Analysis
        yield f"data: {json.dumps({'phase': 'requirements', 'step': 'start', 'message': 'Analyzing requirements against capabilities...'})}\n\n"

        requirement_analysis = []
        for req in rfp["requirements"]:
            await asyncio.sleep(0.3)

            # Match to capabilities
            match_score = 0
            matched_cap = None
            gap = None

            req_lower = req["text"].lower()
            for cap_id, cap in CAPABILITIES.items():
                cap_keywords = cap["name"].lower() + " " + cap["description"].lower()
                if any(word in cap_keywords for word in req_lower.split()[:5]):
                    if cap["maturity"] == "Advanced":
                        match_score = 85 + (hash(req["id"]) % 15)
                    elif cap["maturity"] == "Developing":
                        match_score = 60 + (hash(req["id"]) % 20)
                    else:
                        match_score = 30 + (hash(req["id"]) % 30)
                    matched_cap = cap["name"]
                    if match_score < 70:
                        gap = f"Capability gap: {cap['maturity']} maturity insufficient"
                    break

            if not matched_cap:
                match_score = 20 + (hash(req["id"]) % 20)
                gap = "No matching capability found"

            analysis = {
                "requirement_id": req["id"],
                "type": req["type"],
                "category": req["category"],
                "text": req["text"][:100] + "..." if len(req["text"]) > 100 else req["text"],
                "match_score": match_score,
                "matched_capability": matched_cap,
                "gap": gap
            }
            requirement_analysis.append(analysis)

            yield f"data: {json.dumps({'phase': 'requirements', 'step': 'analyze', 'requirement': analysis})}\n\n"

        # Calculate overall fit
        avg_score = sum(r["match_score"] for r in requirement_analysis) / len(requirement_analysis)
        must_reqs = [r for r in requirement_analysis if r["type"] == "MUST"]
        must_avg = sum(r["match_score"] for r in must_reqs) / len(must_reqs) if must_reqs else 0
        gaps = [r for r in requirement_analysis if r["gap"]]

        yield f"data: {json.dumps({'phase': 'requirements', 'step': 'summary', 'avg_score': round(avg_score, 1), 'must_avg': round(must_avg, 1), 'gap_count': len(gaps)})}\n\n"
        await asyncio.sleep(0.5)

        # Phase 3: LLM Tribunal Voting
        yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'start', 'message': 'Initiating Multi-LLM Tribunal...'})}\n\n"
        await asyncio.sleep(0.3)

        # Build analysis prompt
        system_prompt = """You are an expert RFP analyst for a technology consulting firm.
Analyze the RFP and provide a clear BID or NO BID recommendation with reasoning.
Consider: capability fit, risk factors, competitive positioning, and profitability.
Be concise but thorough. Start your response with your recommendation."""

        analysis_prompt = f"""
Analyze this RFP for bid/no-bid decision:

RFP: {rfp['name']}
Client: {rfp['client']}
Industry: {rfp['industry']}
Value: {rfp['value']}
Deadline: {rfp['deadline']}
Difficulty: {rfp['difficulty']}

Requirements Summary:
{chr(10).join(f"- [{r['type']}] {r['text'][:100]}" for r in rfp['requirements'])}

Our Capability Analysis:
- Average Match Score: {avg_score:.1f}%
- MUST Requirement Match: {must_avg:.1f}%
- Capability Gaps: {len(gaps)}

Key Gaps:
{chr(10).join(f"- {g['text'][:80]}: {g['gap']}" for g in gaps[:3]) if gaps else "None identified"}

Provide your BID or NO BID recommendation with key reasoning.
"""

        votes = []

        if SIMULATION_MODE:
            # SIMULATION MODE - Use mock responses
            # Vote 1: Claude (simulated)
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'voting', 'model': 'claude-sonnet-4', 'status': 'thinking'})}\n\n"
            await asyncio.sleep(1.2)
            mock = MOCK_RESPONSES["claude"]
            vote = {"model": "claude-sonnet-4", "verdict": mock["verdict"], "confidence": mock["confidence"], "rationale": mock["rationale"], "raw_response": mock["rationale"]}
            votes.append(vote)
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'vote', 'vote': vote})}\n\n"

            await asyncio.sleep(0.5)

            # Vote 2: GPT-4 (simulated)
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'voting', 'model': 'gpt-4o', 'status': 'thinking'})}\n\n"
            await asyncio.sleep(1.0)
            mock = MOCK_RESPONSES["gpt4"]
            vote = {"model": "gpt-4o", "verdict": mock["verdict"], "confidence": mock["confidence"], "rationale": mock["rationale"], "raw_response": mock["rationale"]}
            votes.append(vote)
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'vote', 'vote': vote})}\n\n"

            await asyncio.sleep(0.5)

            # Vote 3: Grok (simulated)
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'voting', 'model': 'grok-2', 'status': 'thinking'})}\n\n"
            await asyncio.sleep(0.8)
            mock = MOCK_RESPONSES["grok"]
            vote = {"model": "grok-2", "verdict": mock["verdict"], "confidence": mock["confidence"], "rationale": mock["rationale"], "raw_response": mock["rationale"]}
            votes.append(vote)
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'vote', 'vote': vote})}\n\n"

            await asyncio.sleep(0.5)
        else:
            # LIVE MODE - Call real LLM APIs
            # Vote 1: Claude
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'voting', 'model': 'claude-sonnet-4', 'status': 'thinking'})}\n\n"
            claude_result = await call_claude(analysis_prompt, system_prompt)
            if claude_result.get("success"):
                vote = parse_vote(claude_result["response"])
                vote["model"] = "claude-sonnet-4"
                vote["raw_response"] = claude_result["response"]
                votes.append(vote)
                yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'vote', 'vote': vote})}\n\n"
            else:
                yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'error', 'model': 'claude-sonnet-4', 'error': sanitize_error(str(claude_result.get('error', 'Unknown error')))})}\n\n"

            await asyncio.sleep(0.3)

            # Vote 2: GPT-4
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'voting', 'model': 'gpt-4o', 'status': 'thinking'})}\n\n"
            gpt_result = await call_openai(analysis_prompt, system_prompt)
            if gpt_result.get("success"):
                vote = parse_vote(gpt_result["response"])
                vote["model"] = "gpt-4o"
                vote["raw_response"] = gpt_result["response"]
                votes.append(vote)
                yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'vote', 'vote': vote})}\n\n"
            else:
                yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'error', 'model': 'gpt-4o', 'error': sanitize_error(str(gpt_result.get('error', 'Unknown error')))})}\n\n"

            await asyncio.sleep(0.3)

            # Vote 3: Grok
            yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'voting', 'model': 'grok-2', 'status': 'thinking'})}\n\n"
            grok_result = await call_grok(analysis_prompt, system_prompt)
            if grok_result.get("success"):
                vote = parse_vote(grok_result["response"])
                vote["model"] = "grok-2"
                vote["raw_response"] = grok_result["response"]
                votes.append(vote)
                yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'vote', 'vote': vote})}\n\n"
            else:
                yield f"data: {json.dumps({'phase': 'tribunal', 'step': 'error', 'model': 'grok-2', 'error': sanitize_error(str(grok_result.get('error', 'Unknown error')))})}\n\n"

            await asyncio.sleep(0.5)

        # Phase 4: Consensus Calculation
        yield f"data: {json.dumps({'phase': 'consensus', 'step': 'start', 'message': 'Calculating tribunal consensus...'})}\n\n"
        await asyncio.sleep(0.3)

        if votes:
            # Count verdicts
            verdict_counts = {}
            total_confidence = 0
            for v in votes:
                verdict_counts[v["verdict"]] = verdict_counts.get(v["verdict"], 0) + 1
                total_confidence += v["confidence"]

            # Determine consensus
            max_verdict = max(verdict_counts, key=verdict_counts.get)
            consensus_strength = verdict_counts[max_verdict] / len(votes)
            avg_confidence = total_confidence / len(votes)

            # Calculate phi (agreement), psi (confidence), delta (decisiveness)
            phi = consensus_strength
            psi = avg_confidence
            delta = abs(0.5 - avg_confidence) * 2  # How far from uncertain

            consensus = {
                "verdict": max_verdict,
                "confidence": round(avg_confidence, 3),
                "phi": round(phi, 3),
                "psi": round(psi, 3),
                "delta": round(delta, 3),
                "vote_count": len(votes),
                "unanimous": consensus_strength == 1.0
            }
        else:
            consensus = {
                "verdict": "REVIEW",
                "confidence": 0.5,
                "phi": 0,
                "psi": 0,
                "delta": 0,
                "vote_count": 0,
                "unanimous": False,
                "error": "No LLM votes received - check API keys"
            }

        yield f"data: {json.dumps({'phase': 'consensus', 'step': 'result', 'consensus': consensus})}\n\n"
        await asyncio.sleep(0.5)

        # Phase 5: Final Analysis Report
        yield f"data: {json.dumps({'phase': 'report', 'step': 'start', 'message': 'Generating final analysis report...'})}\n\n"
        await asyncio.sleep(0.3)

        # Risk assessment
        risks = []
        if must_avg < 70:
            risks.append({"level": "HIGH", "description": "Significant gaps in MUST requirements"})
        if len(gaps) > len(requirement_analysis) / 2:
            risks.append({"level": "MEDIUM", "description": "More than 50% of requirements have capability gaps"})
        if rfp["difficulty"] == "Very High":
            risks.append({"level": "HIGH", "description": "Very high complexity project"})
        if rfp["difficulty"] == "High" and avg_score < 60:
            risks.append({"level": "MEDIUM", "description": "High complexity with moderate capability fit"})

        # Opportunities
        opportunities = []
        if avg_score > 75:
            opportunities.append("Strong capability alignment for competitive advantage")
        if rfp["industry"] in ["Healthcare", "Retail"]:
            opportunities.append(f"Established track record in {rfp['industry']} industry")
        if float(rfp["value"].replace("$", "").replace("M", "").replace("K", "")) > 1:
            opportunities.append("High-value engagement with growth potential")

        final_report = {
            "rfp": {
                "id": rfp["id"],
                "name": rfp["name"],
                "client": rfp["client"],
                "industry": rfp["industry"],
                "value": rfp["value"]
            },
            "analysis": {
                "overall_fit": round(avg_score, 1),
                "must_requirement_fit": round(must_avg, 1),
                "gap_count": len(gaps),
                "total_requirements": len(requirement_analysis)
            },
            "tribunal": {
                "verdict": consensus["verdict"],
                "confidence": consensus["confidence"],
                "votes": len(votes),
                "unanimous": consensus.get("unanimous", False)
            },
            "risks": risks,
            "opportunities": opportunities,
            "recommendation": consensus["verdict"],
            "next_steps": [
                "Review detailed capability gaps" if gaps else "Proceed with proposal development",
                "Assess resource availability for timeline",
                "Identify potential teaming partners for gaps" if len(gaps) > 2 else "Develop win themes",
                "Schedule bid/no-bid review meeting"
            ],
            "votes_detail": votes
        }

        yield f"data: {json.dumps({'phase': 'report', 'step': 'complete', 'report': final_report})}\n\n"
        yield f"data: {json.dumps({'phase': 'done', 'message': 'Analysis complete'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ==================== REPORT GENERATION ====================
# In-memory cache for completed analyses (for report generation)
completed_analyses = {}

from report_generator import generate_analysis_report


@app.post("/api/analysis/{rfp_id}/save")
async def save_analysis(rfp_id: str, request: Request):
    """Save analysis results for report generation."""
    data = await request.json()
    completed_analyses[rfp_id] = {
        'rfp_data': data.get('rfp_data', {}),
        'analysis_data': data.get('analysis_data', {}),
        'votes': data.get('votes', []),
        'requirements': data.get('requirements', []),
        'saved_at': datetime.now().isoformat()
    }
    return {"status": "saved", "rfp_id": rfp_id}


@app.get("/api/report/{rfp_id}/{report_type}")
async def generate_report(rfp_id: str, report_type: str):
    """
    Generate a PDF report for a completed analysis.

    Report types:
    - executive: High-level summary for leadership
    - technical: Detailed analysis with math/semantic justification
    - legal: Compliance and risk focused
    - full: Complete tribunal record

    Returns: PDF file download
    """
    # Check if we have stored analysis
    if rfp_id not in completed_analyses:
        # Generate sample data for demo
        rfp = SAMPLE_RFPS.get(rfp_id)
        if not rfp:
            return {"error": "RFP not found"}

        # Create demo analysis data
        analysis_data = {
            'recommendation': 'BID',
            'tribunal': {
                'confidence': 0.82,
                'phi': 0.785,
                'psi': 0.823,
                'delta': 0.038,
                'votes': 3,
                'unanimous': False
            },
            'analysis': {
                'overall_fit': 78,
                'must_requirement_fit': 85,
                'gap_count': 2
            },
            'risks': [
                {'level': 'MEDIUM', 'description': 'Timeline requires careful resource planning'},
                {'level': 'LOW', 'description': 'Some technical clarifications needed'}
            ],
            'next_steps': [
                'Schedule discovery call with client',
                'Prepare technical approach document',
                'Identify project team and availability'
            ]
        }

        rfp_data = {
            'id': rfp_id,
            'name': rfp['name'],
            'client': rfp['client'],
            'industry': rfp.get('industry', 'N/A'),
            'value': rfp['value'],
            'deadline': rfp['deadline'],
            'difficulty': rfp['difficulty']
        }

        votes = [
            {'model': 'claude-sonnet-4', 'verdict': 'BID', 'confidence': 0.85,
             'rationale': 'Strong technical alignment with our capabilities. The requirements match our proven expertise in this domain. Recommend proceeding with bid preparation.'},
            {'model': 'gpt-4o', 'verdict': 'BID', 'confidence': 0.78,
             'rationale': 'Good fit overall. Some gaps identified but manageable with partner support. Timeline is aggressive but achievable.'},
            {'model': 'grok-2', 'verdict': 'CONDITIONAL', 'confidence': 0.72,
             'rationale': 'Technical fit is good but timeline concerns exist. Recommend clarifying scope before full commitment.'}
        ]

        requirements = []
        for i, req in enumerate(rfp.get('requirements', [])[:10], 1):
            requirements.append({
                'requirement_id': f'REQ-{i:03d}',
                'type': req.get('type', 'SHALL'),
                'text': req.get('text', f'Requirement {i}'),
                'match_score': req.get('match_score', 75 + (i % 25)),
                'gap': req.get('gap', None)
            })
    else:
        stored = completed_analyses[rfp_id]
        analysis_data = stored['analysis_data']
        rfp_data = stored['rfp_data']
        votes = stored['votes']
        requirements = stored['requirements']

    # Validate report type
    valid_types = ['executive', 'technical', 'legal', 'full']
    if report_type not in valid_types:
        return {"error": f"Invalid report type. Use: {valid_types}"}

    # Generate PDF
    try:
        pdf_bytes = generate_analysis_report(
            report_type=report_type,
            analysis_data=analysis_data,
            rfp_data=rfp_data,
            votes=votes,
            requirements=requirements
        )

        # Return as downloadable file
        filename = f"RFP_Analysis_{rfp_id}_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Report-Type": report_type,
                "X-RFP-ID": rfp_id
            }
        )
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/report-types")
async def list_report_types():
    """List available report types with descriptions."""
    return {
        "report_types": [
            {
                "id": "executive",
                "name": "Executive Summary",
                "description": "High-level decision summary for leadership and business stakeholders",
                "pages": "2-3",
                "audience": "C-Suite, Business Leaders"
            },
            {
                "id": "technical",
                "name": "Technical Analysis",
                "description": "Detailed analysis with mathematical scoring, semantic matching, and full LLM reasoning",
                "pages": "5-8",
                "audience": "Technical Teams, Solution Architects"
            },
            {
                "id": "legal",
                "name": "Legal & Compliance",
                "description": "Compliance requirements, contract risks, and regulatory considerations",
                "pages": "3-5",
                "audience": "Legal, Compliance, Risk Management"
            },
            {
                "id": "full",
                "name": "Complete Tribunal Record",
                "description": "Full audit trail with all data, votes, and cryptographic verification",
                "pages": "8-12",
                "audience": "Audit, Quality Assurance, Archives"
            }
        ]
    }


@app.get("/demo", response_class=HTMLResponse)
async def demo():
    """Serve the demo UI."""
    # Try index.html first (updated platform), then rfp_demo.html
    for html_name in ["index.html", "rfp_demo.html"]:
        html_path = os.path.join(os.path.dirname(__file__), html_name)
        if os.path.exists(html_path):
            with open(html_path, 'r', encoding='utf-8') as f:
                return f.read()
    return "<h1>Demo UI not found. Run the API and access /demo</h1>"


@app.get("/studio", response_class=HTMLResponse)
async def governance_studio():
    """Serve the Atomic Governance Studio."""
    html_path = os.path.join(os.path.dirname(__file__), "governance-studio.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Governance Studio not found</h1>"


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("RFP INTELLIGENCE PLATFORM - LIVE DEMO")
    print("="*60)
    print("\nLLM Status:")
    print(f"  Claude: {'[OK] Configured' if ANTHROPIC_API_KEY else '[X] Set ANTHROPIC_API_KEY'}")
    print(f"  GPT-4:  {'[OK] Configured' if OPENAI_API_KEY else '[X] Set OPENAI_API_KEY'}")
    print(f"  Grok:   {'[OK] Configured' if XAI_API_KEY else '[X] Set XAI_API_KEY'}")
    print(f"\nStarting server on http://localhost:3020")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=3020)
