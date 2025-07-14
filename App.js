import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, Activity, Droplets, AlertTriangle, Send, Bot, User, 
  Monitor, Target, Scan, Brain, Cpu, Network, Loader, RotateCcw,
  Eye, Zap, Thermometer, FlaskConical, Wind, CheckCircle, XCircle, Wifi, WifiOff
} from 'lucide-react';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_CV_BACKEND_URL || 'http://localhost:8000';

// Backend API Client
class CVBackendClient {
  constructor(baseURL = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  async queryMaskRCNN(cameraId, parameters = {}) {
    try {
      const requestBody = {
        camera_id: cameraId,
        analysis_type: parameters.detailed_analysis ? 'detailed' : 
                      parameters.quick_scan ? 'quick' : 'standard',
        confidence_threshold: parameters.confidence_threshold || 0.7,
        nms_threshold: parameters.nms_threshold || 0.5,
        max_detections: parameters.max_detections || 100,
        object_classes: parameters.object_classes || null
      };

      const response = await fetch(`${this.baseURL}/api/cv/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`CV API error: ${response.status}`);
      }

      const result = await response.json();
      
      return {
        detections: result.detections || [],
        processing_time: result.processing_time * 1000,
        total_objects: result.total_objects,
        timestamp: new Date(result.timestamp),
        success: result.success,
        error_message: result.error_message
      };

    } catch (error) {
      console.error('Mask R-CNN query failed:', error);
      return {
        detections: [],
        processing_time: 0,
        total_objects: 0,
        timestamp: new Date(),
        success: false,
        error_message: error.message
      };
    }
  }

  async parseLLMIntent(userInput, context = {}) {
    try {
      const response = await fetch(`${this.baseURL}/api/llm/parse-intent`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_input: userInput,
          context: context
        })
      });

      if (!response.ok) {
        throw new Error(`LLM parsing error: ${response.status}`);
      }

      return await response.json();

    } catch (error) {
      console.error('LLM parsing failed:', error);
      return {
        action: 'none',
        target_cameras: [],
        object_classes: [],
        analysis_type: 'standard',
        priority: 'normal',
        parameters: {},
        confidence: 0.3,
        reasoning: `LLM parsing failed: ${error.message}`
      };
    }
  }

  async getSystemStatus() {
    try {
      const response = await fetch(`${this.baseURL}/api/system/status`);
      if (!response.ok) throw new Error('Failed to fetch status');
      return await response.json();
    } catch (error) {
      return { status: 'error', error: error.message };
    }
  }

  async getCameras() {
    try {
      const response = await fetch(`${this.baseURL}/api/cameras/list`);
      if (!response.ok) throw new Error('Failed to fetch cameras');
      const result = await response.json();
      return result.cameras;
    } catch (error) {
      console.error('Failed to get cameras:', error);
      return [];
    }
  }

  async healthCheck() {
    try {
      const response = await fetch(`${this.baseURL}/`);
      return response.ok;
    } catch (error) {
      return false;
    }
  }
}

const WaterTreatmentCVFrontend = () => {
  // State management
  const [backendClient] = useState(() => new CVBackendClient());
  const [isConnected, setIsConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedCamera, setSelectedCamera] = useState('main-tank');
  const [cvProcessing, setCvProcessing] = useState(false);
  const [llmProcessing, setLlmProcessing] = useState(false);
  
  // Chat state
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Water Treatment CV System online! Connected to real backend with Mask R-CNN and LLM capabilities. Try saying "scan the main tank for sediment" or "check all cameras for particles".',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const chatEndRef = useRef(null);

  // Sensor data (simulated for frontend)
  const [sensorData, setSensorData] = useState({
    ph: 7.3,
    turbidity: 2.1,
    chlorine: 1.8,
    temperature: 23.2,
    flowRate: 1340,
    dissolvedOxygen: 8.2
  });

  // Detection results from backend
  const [detectionResults, setDetectionResults] = useState({});

  // Available cameras
  const [cameras, setCameras] = useState([
    { id: 'main-tank', name: 'Main Treatment Tank', status: 'active' },
    { id: 'aeration', name: 'Aeration Basin', status: 'active' },
    { id: 'filtration', name: 'Filtration System', status: 'active' },
    { id: 'outflow', name: 'Outflow Monitor', status: 'active' }
  ]);

  // Backend connectivity check
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const isHealthy = await backendClient.healthCheck();
        setIsConnected(isHealthy);
        
        if (isHealthy) {
          const status = await backendClient.getSystemStatus();
          setSystemStatus(status);
          
          const availableCameras = await backendClient.getCameras();
          if (availableCameras.length > 0) {
            setCameras(availableCameras);
          }
        }
      } catch (error) {
        setIsConnected(false);
        setSystemStatus({ status: 'error', error: error.message });
      }
    };

    checkConnection();
    const interval = setInterval(checkConnection, 10000); // Check every 10 seconds
    return () => clearInterval(interval);
  }, [backendClient]);

  // Auto-update sensor data
  useEffect(() => {
    const interval = setInterval(() => {
      setSensorData(prev => ({
        ...prev,
        ph: +(prev.ph + (Math.random() - 0.5) * 0.05).toFixed(2),
        turbidity: +(Math.max(0, prev.turbidity + (Math.random() - 0.5) * 0.1)).toFixed(1),
        chlorine: +(prev.chlorine + (Math.random() - 0.5) * 0.05).toFixed(2),
        temperature: +(prev.temperature + (Math.random() - 0.5) * 0.3).toFixed(1),
        flowRate: Math.round(prev.flowRate + (Math.random() - 0.5) * 30)
      }));
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Main message handler with real backend integration
  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return;

    const userMessage = { 
      role: 'user', 
      content: inputMessage,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    const originalMessage = inputMessage;
    setInputMessage('');
    setIsProcessing(true);

    if (!isConnected) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: '❌ Backend is not connected. Please check if the backend server is running on http://localhost:8000',
        timestamp: new Date()
      }]);
      setIsProcessing(false);
      return;
    }

    try {
      // Step 1: Parse intent using real backend LLM
      setLlmProcessing(true);
      const cvIntent = await backendClient.parseLLMIntent(originalMessage, {
        available_cameras: cameras.map(c => c.id),
        current_camera: selectedCamera,
        sensor_data: sensorData
      });
      setLlmProcessing(false);
      
      // Step 2: Show LLM parsing results
      if (cvIntent.confidence > 0.5 && cvIntent.action === 'cv_query') {
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `🧠 LLM Analysis: ${cvIntent.reasoning} (Confidence: ${(cvIntent.confidence * 100).toFixed(1)}%)`,
          timestamp: new Date()
        }]);
      }

      // Step 3: Execute CV queries based on LLM understanding
      if (cvIntent.action === 'cv_query' && cvIntent.target_cameras.length > 0) {
        const cameraList = cvIntent.target_cameras.join(', ');
        const objectFocus = cvIntent.object_classes.length > 0 ? 
          ` focusing on ${cvIntent.object_classes.join(', ')}` : '';
        
        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `🎯 Executing real Mask R-CNN analysis on ${cameraList}${objectFocus}...`,
          timestamp: new Date()
        }]);

        setCvProcessing(true);

        // Execute real CV queries on target cameras
        for (const cameraId of cvIntent.target_cameras) {
          if (cameras.find(c => c.id === cameraId && c.status === 'active')) {
            const result = await backendClient.queryMaskRCNN(cameraId, {
              detailed_analysis: cvIntent.analysis_type === 'detailed',
              quick_scan: cvIntent.analysis_type === 'quick',
              object_classes: cvIntent.object_classes.length > 0 ? cvIntent.object_classes : null,
              confidence_threshold: 0.7
            });

            // Update detection results with real backend data
            setDetectionResults(prev => ({
              ...prev,
              [cameraId]: {
                detections: result.detections,
                processing_time: result.processing_time,
                total_objects: result.total_objects,
                timestamp: result.timestamp,
                success: result.success,
                error_message: result.error_message
              }
            }));
          }
        }
        setCvProcessing(false);
        await new Promise(resolve => setTimeout(resolve, 1000));
      }

      // Step 4: Generate intelligent response based on real results
      setTimeout(() => {
        const primaryCamera = cvIntent.target_cameras[0] || selectedCamera;
        const currentResults = detectionResults[primaryCamera];
        
        let response = '';
        if (cvIntent.action === 'cv_query' && currentResults) {
          if (cvIntent.analysis_type === 'counting') {
            const objectCounts = cvIntent.object_classes.length > 0 ? 
              cvIntent.object_classes.map(cls => {
                const count = currentResults.detections?.filter(d => d.class_name === cls).length || 0;
                return `${count} ${cls}s`;
              }).join(', ') :
              `${currentResults.total_objects || 0} total objects`;
            
            response = `📊 Real CV Count Analysis: ${objectCounts} detected in ${primaryCamera}`;
            
          } else if (cvIntent.analysis_type === 'comprehensive') {
            const totalObjects = cvIntent.target_cameras.reduce((sum, camId) => 
              sum + (detectionResults[camId]?.total_objects || 0), 0);
            response = `🔬 Comprehensive Real Analysis: ${totalObjects} objects across ${cvIntent.target_cameras.length} cameras. Backend processing complete.`;
            
          } else {
            const detections = currentResults.detections || [];
            const bubbles = detections.filter(d => d.class_name === 'bubble').length;
            const particles = detections.filter(d => d.class_name === 'particle').length;
            const avgConfidence = detections.length > 0 ? 
              detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length : 0;
            
            response = `✅ Real Backend Analysis Complete: ${currentResults.total_objects || 0} objects in ${primaryCamera}. ${bubbles} bubbles, ${particles} particles. Confidence: ${(avgConfidence * 100).toFixed(1)}%. Processing: ${Math.round(currentResults.processing_time || 0)}ms`;
          }
          
          if (!currentResults.success) {
            response += ` ⚠️ Error: ${currentResults.error_message}`;
          }
          
        } else if (originalMessage.toLowerCase().includes('status') || originalMessage.toLowerCase().includes('health')) {
          response = `🔧 System Status: Backend ${isConnected ? 'Connected' : 'Disconnected'}. ${systemStatus?.status || 'Unknown'} status. ${cameras.filter(c => c.status === 'active').length} cameras active.`;
          
        } else {
          response = `🤖 Backend Connected: Real CV system ready. Try natural language like "scan main tank for sediment" or "analyze all cameras quickly". LLM parsing confidence: ${(cvIntent.confidence * 100).toFixed(1)}%`;
        }

        setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: response,
          timestamp: new Date()
        }]);
        setIsProcessing(false);
      }, 1500);
      
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: `❌ Backend Error: ${error.message}. Check backend connectivity.`,
        timestamp: new Date()
      }]);
      setIsProcessing(false);
      setLlmProcessing(false);
      setCvProcessing(false);
    }
  };

  // Manual CV query trigger
  const handleManualCVQuery = async (cameraId) => {
    if (!isConnected) {
      alert('Backend not connected!');
      return;
    }

    setCvProcessing(true);
    try {
      const result = await backendClient.queryMaskRCNN(cameraId);
      setDetectionResults(prev => ({
        ...prev,
        [cameraId]: result
      }));
    } catch (error) {
      console.error('Manual CV query failed:', error);
    } finally {
      setCvProcessing(false);
    }
  };

  // Render connection status
  const renderConnectionStatus = () => (
    <div className={`flex items-center space-x-2 ${
      isConnected ? 'text-green-400' : 'text-red-400'
    }`}>
      {isConnected ? <Wifi className="h-4 w-4" /> : <WifiOff className="h-4 w-4" />}
      <span className="text-sm">
        Backend {isConnected ? 'Connected' : 'Disconnected'}
      </span>
      {isConnected && systemStatus?.status === 'healthy' && (
        <CheckCircle className="h-4 w-4 text-green-400" />
      )}
    </div>
  );

  // Render backend status panel
  const renderBackendStatus = () => (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-4 mb-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Network className="h-5 w-5 text-blue-400" />
          <h3 className="font-semibold">Backend Connection</h3>
        </div>
        {renderConnectionStatus()}
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div className="text-center">
          <div className={`font-bold ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
            {isConnected ? 'Online' : 'Offline'}
          </div>
          <div className="text-slate-400">Status</div>
        </div>
        <div className="text-center">
          <div className="text-blue-400 font-bold">
            {API_BASE_URL.replace('http://', '')}
          </div>
          <div className="text-slate-400">Endpoint</div>
        </div>
        <div className="text-center">
          <div className="text-purple-400 font-bold">
            {systemStatus?.llm_available ? 'LLM Ready' : 'No LLM'}
          </div>
          <div className="text-slate-400">Natural Language</div>
        </div>
        <div className="text-center">
          <div className="text-cyan-400 font-bold">
            {systemStatus?.model_loaded ? 'CV Ready' : 'No Model'}
          </div>
          <div className="text-slate-400">Computer Vision</div>
        </div>
      </div>
      
      {!isConnected && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <div className="flex items-center space-x-2">
            <XCircle className="h-4 w-4 text-red-400" />
            <span className="text-sm text-red-400">
              Backend not reachable. Make sure backend is running: `uvicorn main:app --reload`
            </span>
          </div>
        </div>
      )}
    </div>
  );

  // Render sensor grid
  const renderSensorGrid = () => (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
      {[
        { label: 'pH', value: sensorData.ph, unit: '', icon: FlaskConical },
        { label: 'Turbidity', value: sensorData.turbidity, unit: 'NTU', icon: Eye },
        { label: 'Chlorine', value: sensorData.chlorine, unit: 'mg/L', icon: Zap },
        { label: 'Temperature', value: sensorData.temperature, unit: '°C', icon: Thermometer },
        { label: 'Flow Rate', value: sensorData.flowRate, unit: 'L/min', icon: Wind },
        { label: 'DO', value: sensorData.dissolvedOxygen, unit: 'mg/L', icon: Activity }
      ].map((sensor) => (
        <div key={sensor.label} className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <sensor.icon className="h-5 w-5 text-blue-400" />
            <span className="text-xs text-slate-400">{sensor.label}</span>
          </div>
          <div className="text-lg font-bold text-white">
            {sensor.value}{sensor.unit}
          </div>
        </div>
      ))}
    </div>
  );

  // Render camera grid with real detection results
  const renderCameraGrid = () => (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      {cameras.map((camera) => {
        const cameraResults = detectionResults[camera.id];
        
        return (
          <div 
            key={camera.id}
            className={`bg-slate-800/50 backdrop-blur-sm border rounded-xl p-6 cursor-pointer transition-all ${
              selectedCamera === camera.id 
                ? 'border-blue-500 ring-2 ring-blue-500/20' 
                : 'border-slate-700 hover:border-slate-600'
            }`}
            onClick={() => setSelectedCamera(camera.id)}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Camera className="h-5 w-5 text-blue-400" />
                <span className="font-medium">{camera.name}</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  camera.status === 'active' ? 'bg-green-400' : 'bg-red-400'
                }`}></div>
                <button 
                  onClick={(e) => {
                    e.stopPropagation();
                    handleManualCVQuery(camera.id);
                  }}
                  disabled={cvProcessing || !isConnected}
                  className="p-1 bg-blue-600 rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  {cvProcessing ? 
                    <Loader className="h-3 w-3 animate-spin" /> : 
                    <Scan className="h-3 w-3" />
                  }
                </button>
              </div>
            </div>
            
            {/* Camera Feed Simulation */}
            <div className="aspect-video bg-gradient-to-br from-slate-900 to-slate-700 rounded-lg relative overflow-hidden mb-4">
              {/* Real Detection Overlays */}
              <div className="absolute inset-0">
                {cameraResults?.detections?.map((detection, idx) => (
                  <div
                    key={idx}
                    className={`absolute border-2 rounded ${
                      detection.class_name === 'bubble' ? 'border-blue-400' :
                      detection.class_name === 'particle' ? 'border-yellow-400' :
                      detection.class_name === 'sediment' ? 'border-orange-400' :
                      detection.class_name === 'foam' ? 'border-white' :
                      'border-red-400'
                    }`}
                    style={{
                      left: `${(detection.bbox[0] / 640) * 100}%`,
                      top: `${(detection.bbox[1] / 480) * 100}%`,
                      width: `${(detection.bbox[2] / 640) * 100}%`,
                      height: `${(detection.bbox[3] / 480) * 100}%`,
                    }}
                  >
                    <div className={`absolute -top-6 left-0 text-xs px-1 rounded ${
                      detection.class_name === 'bubble' ? 'bg-blue-400' :
                      detection.class_name === 'particle' ? 'bg-yellow-400' :
                      detection.class_name === 'sediment' ? 'bg-orange-400' :
                      detection.class_name === 'foam' ? 'bg-white' :
                      'bg-red-400'
                    } text-black`}>
                      {detection.class_name} {(detection.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="absolute top-2 left-2 text-xs bg-blue-600/80 px-2 py-1 rounded">
                {cameraResults?.total_objects || 0} objects detected
              </div>
              <div className="absolute bottom-2 right-2 text-xs bg-black/50 px-2 py-1 rounded">
                {isConnected ? 'Real Backend' : 'Disconnected'}
              </div>
              {cameraResults?.processing_time && (
                <div className="absolute bottom-2 left-2 text-xs bg-green-600/80 px-2 py-1 rounded">
                  {Math.round(cameraResults.processing_time)}ms
                </div>
              )}
            </div>

            {/* Detection Results */}
            <div className="space-y-2">
              {cameraResults?.detections?.map((detection, idx) => (
                <div key={idx} className="flex items-center justify-between p-2 bg-slate-900/50 rounded">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${
                      detection.class_name === 'bubble' ? 'bg-blue-400' :
                      detection.class_name === 'particle' ? 'bg-yellow-400' :
                      detection.class_name === 'sediment' ? 'bg-orange-400' :
                      detection.class_name === 'foam' ? 'bg-white' :
                      'bg-red-400'
                    }`}></div>
                    <span className="text-sm capitalize">{detection.class_name}</span>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-green-400">
                      {(detection.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-slate-400">
                      Area: {detection.mask_area}px²
                    </div>
                  </div>
                </div>
              )) || (
                <div className="text-center text-slate-400 py-4">
                  No detections yet - click scan button
                </div>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 text-white">
      {/* Header */}
      <div className="border-b border-slate-700 bg-slate-800/50 backdrop-blur-sm">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg">
              <Monitor className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold">Real CV + LLM Frontend</h1>
              <p className="text-sm text-slate-400">Connected to FastAPI Backend</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex space-x-2">
              {['overview', 'cameras'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1 rounded-lg text-sm transition-colors ${
                    activeTab === tab 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              ))}
            </div>
            {renderConnectionStatus()}
          </div>
        </div>
      </div>

      <div className="flex h-[calc(100vh-80px)]">
        {/* Main Content */}
        <div className="flex-1 p-6 overflow-y-auto">
          {renderBackendStatus()}
          
          {activeTab === 'overview' && (
            <>
              {renderSensorGrid()}
              {renderCameraGrid()}
            </>
          )}

          {activeTab === 'cameras' && (
            <>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-semibold">Real-time Camera Analysis</h2>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-slate-400">
                    Backend: {isConnected ? 'Connected' : 'Offline'}
                  </span>
                </div>
              </div>
              {renderCameraGrid()}
            </>
          )}
        </div>

        {/* AI Chat Interface */}
        <div className="w-96 border-l border-slate-700 bg-slate-800/30 backdrop-blur-sm flex flex-col">
          <div className="p-4 border-b border-slate-700">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-blue-400" />
              <h3 className="font-semibold">Real LLM + CV Assistant</h3>
            </div>
            <p className="text-xs text-slate-400 mt-1">Natural language → Real backend</p>
            {(llmProcessing || cvProcessing) && (
              <div className="flex items-center space-x-2 mt-2 text-yellow-400">
                <Loader className="h-3 w-3 animate-spin" />
                <span className="text-xs">
                  {llmProcessing ? 'LLM processing...' : 'CV analyzing...'}
                </span>
              </div>
            )}
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] p-3 rounded-lg ${
                  message.role === 'user' 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-slate-700 text-slate-100'
                }`}>
                  <div className="flex items-start space-x-2">
                    {message.role === 'assistant' && <Brain className="h-4 w-4 mt-0.5 text-blue-400 flex-shrink-0" />}
                    {message.role === 'user' && <User className="h-4 w-4 mt-0.5 text-blue-200 flex-shrink-0" />}
                    <div>
                      <p className="text-sm leading-relaxed">{message.content}</p>
                      <p className="text-xs opacity-60 mt-1">
                        {message.timestamp.toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {isProcessing && (
              <div className="flex justify-start">
                <div className="bg-slate-700 p-3 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Brain className="h-4 w-4 text-blue-400" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <div className="p-4 border-t border-slate-700">
            <div className="flex space-x-2 mb-3">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder={isConnected ? "Try: 'scan main tank for sediment'" : "Backend disconnected..."}
                className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={isProcessing || !isConnected}
              />
              <button
                onClick={handleSendMessage}
                disabled={isProcessing || !inputMessage.trim() || !isConnected}
                className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
            
            <div className="grid grid-cols-1 gap-1 text-xs">
              {[
                'Scan main tank for sediment particles',
                'Check bubble activity in aeration system', 
                'Analyze all cameras for contaminants',
                'System status and backend health'
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInputMessage(suggestion)}
                  disabled={!isConnected}
                  className="bg-slate-700 hover:bg-slate-600 px-2 py-1 rounded transition-colors text-left disabled:opacity-50"
                >
                  {suggestion}
                </button>
              ))}
            </div>
            
            {!isConnected && (
              <div className="mt-2 text-center text-xs text-red-400">
                Start backend: uvicorn main:app --reload
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default WaterTreatmentCVFrontend;


/* 
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Edit <code>src/App.js</code> and save to reload.
        </p>
        <a
          className="App-link"
          href="https://reactjs.org"
          target="_blank"
          rel="noopener noreferrer"
        >
          Learn React
        </a>
      </header>
    </div>
  );
}

export default App;
 */