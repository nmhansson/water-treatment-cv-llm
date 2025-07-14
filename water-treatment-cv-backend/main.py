# Complete Water Treatment CV Backend - All Endpoints Included
# This version has ALL required endpoints for the React frontend

import os
import json
import logging
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Complete Water Treatment CV Backend",
    description="Full CV API with all required endpoints",
    version="2.3.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# PYDANTIC MODELS
# ================================

class CVRequest(BaseModel):
    camera_id: str
    analysis_type: str = "standard"
    confidence_threshold: float = 0.7
    nms_threshold: float = 0.5
    max_detections: int = 100
    object_classes: Optional[List[str]] = None

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]  # [x, y, width, height]
    mask_area: int
    centroid: List[float]  # [x, y]

class CVResponse(BaseModel):
    camera_id: str
    timestamp: datetime
    detections: List[Detection]
    total_objects: int
    processing_time: float
    model_info: Dict[str, Any]
    analysis_type: str
    success: bool
    error_message: Optional[str] = None

class LLMParseRequest(BaseModel):
    user_input: str
    context: Dict[str, Any] = Field(default_factory=dict)

class LLMParseResponse(BaseModel):
    action: str
    target_cameras: List[str]
    object_classes: List[str]
    analysis_type: str
    priority: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str

# ================================
# ANTHROPIC LLM PARSER
# ================================

class AnthropicParser:
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        self.anthropic_available = False
        self.model_name = "claude-3-5-sonnet-20241022"
        
        if not self.api_key or self.api_key in ["your_anthropic_api_key_here", "sk-ant-your-actual-key-here"]:
            logger.warning("❌ ANTHROPIC_API_KEY not configured properly")
            return
            
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.anthropic_available = True
            logger.info(f"✅ Anthropic client initialized with model: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Anthropic client: {e}")
            self.client = None

    async def parse_cv_intent(self, user_input: str, context: Dict[str, Any] = {}) -> LLMParseResponse:
        """Parse using Claude API"""
        
        if not self.anthropic_available:
            logger.info("🔄 Using fallback parsing (Anthropic unavailable)")
            return await self._fallback_parse(user_input)

        try:
            system_prompt = """You are an expert water treatment computer vision system parser. Convert natural language requests into structured CV commands.

Available cameras: main-tank, aeration, filtration, outflow
Available objects: bubble, particle, sediment, foam, equipment, contaminant
Available types: standard, detailed, quick, counting, comprehensive

Respond with valid JSON only:
{
  "action": "cv_query" | "status_check" | "none",
  "target_cameras": ["camera_id"],
  "object_classes": ["class_name"], 
  "analysis_type": "standard" | "detailed" | "quick" | "counting" | "comprehensive",
  "priority": "normal" | "high" | "low",
  "parameters": {},
  "confidence": 0.0-1.0,
  "reasoning": "explanation"
}"""

            logger.info(f"🤖 Calling Claude: {user_input[:50]}...")
            
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model_name,
                max_tokens=1000,
                temperature=0.1,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Parse: '{user_input}'\nContext: {json.dumps(context, default=str)}"
                }]
            )

            response_text = response.content[0].text.strip()
            
            # Clean JSON formatting
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].strip()

            # Parse JSON
            try:
                parsed_data = json.loads(response_text)
                logger.info(f"✅ Claude response parsed: {parsed_data.get('action')} -> {parsed_data.get('target_cameras')}")
                return LLMParseResponse(**parsed_data)
                
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse Claude JSON: {e}")
                return await self._fallback_parse(user_input)

        except Exception as e:
            logger.error(f"❌ Anthropic API error: {e}")
            return await self._fallback_parse(user_input)

    async def _fallback_parse(self, user_input: str) -> LLMParseResponse:
        """Rule-based fallback parsing"""
        lower_input = user_input.lower()
        
        intent = {
            "action": "none",
            "target_cameras": [],
            "object_classes": [],
            "analysis_type": "standard",
            "priority": "normal", 
            "parameters": {},
            "confidence": 0.75,
            "reasoning": "Rule-based parsing (Anthropic API unavailable)"
        }

        # Action detection
        if any(word in lower_input for word in ['scan', 'analyze', 'check', 'detect']):
            intent["action"] = "cv_query"

        # Camera detection
        if 'main' in lower_input: intent["target_cameras"].append('main-tank')
        if 'aeration' in lower_input: intent["target_cameras"].append('aeration')
        if 'filter' in lower_input: intent["target_cameras"].append('filtration')
        if 'outflow' in lower_input: intent["target_cameras"].append('outflow')
        if 'all' in lower_input: intent["target_cameras"] = ['main-tank', 'aeration', 'filtration', 'outflow']

        # Object detection
        for obj in ['bubble', 'particle', 'sediment', 'foam', 'contaminant']:
            if obj in lower_input: intent["object_classes"].append(obj)

        # Default camera
        if intent["action"] == "cv_query" and not intent["target_cameras"]:
            intent["target_cameras"] = ["main-tank"]

        return LLMParseResponse(**intent)

# ================================
# CV DETECTOR
# ================================

class CVDetector:
    def __init__(self):
        self.class_names = ["bubble", "particle", "sediment", "foam", "equipment", "contaminant"]
        logger.info("CV detector initialized")

    def predict(self, camera_id: str, config: CVRequest):
        """Simulate CV detection"""
        start_time = datetime.now()
        
        detections = []
        
        if camera_id == "main-tank":
            detections = [
                Detection(
                    class_name="bubble",
                    confidence=0.94,
                    bbox=[120, 80, 45, 35],
                    mask_area=1250,
                    centroid=[142.5, 97.5]
                ),
                Detection(
                    class_name="sediment", 
                    confidence=0.91,
                    bbox=[50, 300, 80, 60],
                    mask_area=2100,
                    centroid=[90.0, 330.0]
                ),
                Detection(
                    class_name="particle",
                    confidence=0.87,
                    bbox=[200, 150, 25, 20],
                    mask_area=380,
                    centroid=[212.5, 160.0]
                )
            ]
        elif camera_id == "aeration":
            detections = [
                Detection(
                    class_name="bubble",
                    confidence=0.96,
                    bbox=[80, 60, 200, 180], 
                    mask_area=15600,
                    centroid=[180.0, 150.0]
                ),
                Detection(
                    class_name="foam",
                    confidence=0.83,
                    bbox=[220, 20, 60, 30],
                    mask_area=1800,
                    centroid=[250.0, 35.0]
                )
            ]
        elif camera_id == "filtration":
            detections = [
                Detection(
                    class_name="particle",
                    confidence=0.92,
                    bbox=[150, 120, 30, 25],
                    mask_area=560,
                    centroid=[165.0, 132.5]
                ),
                Detection(
                    class_name="contaminant",
                    confidence=0.83,
                    bbox=[180, 180, 15, 12],
                    mask_area=145,
                    centroid=[187.5, 186.0]
                )
            ]
        elif camera_id == "outflow":
            detections = [
                Detection(
                    class_name="particle",
                    confidence=0.88,
                    bbox=[100, 100, 20, 15],
                    mask_area=200,
                    centroid=[110.0, 107.5]
                )
            ]
        
        # Apply filters
        if config.object_classes:
            detections = [d for d in detections if d.class_name in config.object_classes]
        
        detections = [d for d in detections if d.confidence >= config.confidence_threshold]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "detections": detections,
            "total_objects": len(detections),
            "processing_time": processing_time,
            "success": True,
            "model_info": {
                "backbone": "Simulated-ResNet-50",
                "framework": "Testing",
                "classes": self.class_names,
                "device": "cpu"
            }
        }

# Initialize components
detector = CVDetector()
llm_parser = AnthropicParser()

# ================================
# ALL API ENDPOINTS
# ================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Complete Water Treatment CV Backend",
        "status": "online",
        "claude_model": llm_parser.model_name,
        "anthropic_available": llm_parser.anthropic_available,
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/",
            "/api/cv/analyze",
            "/api/llm/parse-intent", 
            "/api/cameras/list",
            "/api/model/info",
            "/api/system/status",
            "/api/anthropic/test"
        ]
    }

@app.post("/api/cv/analyze", response_model=CVResponse)
async def analyze_camera(request: CVRequest):
    """Main CV analysis endpoint"""
    logger.info(f"CV analysis request for camera: {request.camera_id}")
    
    try:
        result = detector.predict(request.camera_id, request)
        
        response = CVResponse(
            camera_id=request.camera_id,
            timestamp=datetime.now(),
            detections=result["detections"],
            total_objects=result["total_objects"],
            processing_time=result["processing_time"],
            model_info=result["model_info"],
            analysis_type=request.analysis_type,
            success=result["success"],
            error_message=result.get("error_message")
        )
        
        logger.info(f"Analysis complete: {result['total_objects']} objects detected")
        return response
        
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/llm/parse-intent", response_model=LLMParseResponse)
async def parse_cv_intent(request: LLMParseRequest):
    """Parse natural language using Claude"""
    logger.info(f"🤖 LLM parse request: {request.user_input[:50]}...")
    
    try:
        result = await llm_parser.parse_cv_intent(request.user_input, request.context)
        logger.info(f"✅ Parse result: {result.action} -> {result.target_cameras}")
        return result
        
    except Exception as e:
        logger.error(f"❌ LLM parsing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cameras/list")
async def list_cameras():
    """List available cameras - REQUIRED ENDPOINT"""
    return {
        "cameras": [
            {"id": "main-tank", "name": "Main Treatment Tank", "status": "active"},
            {"id": "aeration", "name": "Aeration Basin", "status": "active"},
            {"id": "filtration", "name": "Filtration System", "status": "active"},
            {"id": "outflow", "name": "Outflow Monitor", "status": "active"}
        ]
    }

@app.get("/api/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "Simulated Mask R-CNN",
        "backbone": "Simulated-ResNet-50",
        "framework": "Testing",
        "classes": detector.class_names,
        "num_classes": len(detector.class_names),
        "device": "cpu",
        "version": "2.3.0-complete",
        "llm_model": llm_parser.model_name
    }

@app.get("/api/system/status")
async def system_status():
    """System health check - REQUIRED ENDPOINT"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": False,
        "gpu_count": 0,
        "model_loaded": True,
        "llm_available": llm_parser.anthropic_available,
        "llm_type": "anthropic-claude" if llm_parser.anthropic_available else "fallback",
        "claude_model": llm_parser.model_name,
        "api_key_configured": bool(llm_parser.api_key and not llm_parser.api_key.startswith("your")),
        "version": "2.3.0-complete"
    }

@app.get("/api/anthropic/test")
async def test_anthropic():
    """Test Anthropic API connection"""
    if not llm_parser.anthropic_available:
        return {
            "status": "error",
            "message": "Anthropic not available",
            "model": llm_parser.model_name,
            "api_key_configured": bool(llm_parser.api_key)
        }
    
    try:
        test_result = await llm_parser.parse_cv_intent("test connection")
        return {
            "status": "success",
            "message": "Anthropic API working",
            "model": llm_parser.model_name,
            "test_result": {
                "action": test_result.action,
                "confidence": test_result.confidence,
                "reasoning": test_result.reasoning
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Anthropic API test failed: {str(e)}",
            "model": llm_parser.model_name
        }

@app.post("/api/cv/batch-analyze")
async def batch_analyze(camera_ids: List[str], config: CVRequest):
    """Analyze multiple cameras at once"""
    results = {}
    
    for camera_id in camera_ids:
        config.camera_id = camera_id
        try:
            result = detector.predict(camera_id, config)
            results[camera_id] = {
                "detections": result["detections"],
                "total_objects": result["total_objects"],
                "processing_time": result["processing_time"],
                "success": result["success"]
            }
        except Exception as e:
            results[camera_id] = {"error": str(e), "success": False}
    
    return {"results": results, "timestamp": datetime.now().isoformat()}

# ================================
# STARTUP EVENT
# ================================

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting Complete Water Treatment CV Backend...")
    logger.info(f"🤖 Claude model: {llm_parser.model_name}")
    logger.info(f"🔑 Anthropic available: {llm_parser.anthropic_available}")
    logger.info("✅ All endpoints initialized successfully!")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")