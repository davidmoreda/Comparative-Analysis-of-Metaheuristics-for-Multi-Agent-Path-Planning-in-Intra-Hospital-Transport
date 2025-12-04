#!/usr/bin/env python3
"""
Professional Web Frontend for NSGA-II Hospital Path Planning
Flask-based application with modern glassmorphic design
"""

import sys
import pathlib
import io
import base64
import json
from flask import Flask, render_template_string, jsonify, request, session, send_file
from PIL import Image
import numpy as np

# =============================================================
# PATHS
# =============================================================
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MAP_PATH = ROOT_DIR / "data" / "Mapa.bmp"

# Import existing algorithm functions
from algorithms.ga_core import nearest_free_black
from algorithms.ga_runner_multi import run_ga_multi_streamlit

# =============================================================
# FLASK APP SETUP
# =============================================================
app = Flask(__name__)
app.secret_key = 'hospital-path-planning-secret-key-2024'

# Load map globally
img_raw = Image.open(MAP_PATH).convert("L")
env = 255 - np.array(img_raw, dtype=np.uint8)  # free=0, wall=255
H, W = env.shape

# Constants
N_AGENTS = 4
STAGES = ["start", "pickup", "drop"]
AGENT_COLORS = ["#ff4444", "#00ffff", "#ffff00", "#00ff00"]

# =============================================================
# HELPER FUNCTIONS
# =============================================================

def init_session():
    """Initialize session state"""
    if 'agent' not in session:
        session['agent'] = 0
    if 'stage' not in session:
        session['stage'] = 0
    if 'points' not in session:
        session['points'] = {str(a): {} for a in range(N_AGENTS)}
    if 'routes' not in session:
        session['routes'] = None

def get_current_stage_name():
    """Get current stage name"""
    return STAGES[session['stage']]

def get_current_agent():
    """Get current agent index"""
    return session['agent']

# =============================================================
# API ROUTES
# =============================================================

@app.route('/')
def index():
    """Main page"""
    init_session()
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/map')
def get_map():
    """Serve the hospital map image"""
    return send_file(MAP_PATH, mimetype='image/bmp')

@app.route('/api/map-hires')
def get_map_hires():
    """Serve high-resolution map for Plotly (4x scale)"""
    # Load original image
    img = Image.open(MAP_PATH).convert('RGB')
    # Scale up 4x with NEAREST for crisp pixels
    new_size = (img.width * 4, img.height * 4)
    img_scaled = img.resize(new_size, Image.NEAREST)
    
    # Save to bytes
    img_io = io.BytesIO()
    img_scaled.save(img_io, 'PNG', optimize=False)
    img_io.seek(0)
    
    return send_file(img_io, mimetype='image/png')

@app.route('/api/state')
def get_state():
    """Get current application state"""
    init_session()
    return jsonify({
        'agent': session['agent'],
        'stage': session['stage'],
        'stage_name': get_current_stage_name(),
        'points': session['points'],
        'routes': session['routes'],
        'map_size': {'width': W, 'height': H}
    })

@app.route('/api/select-point', methods=['POST'])
def select_point():
    """Validate and store a selected point"""
    init_session()
    
    data = request.json
    x_raw = int(data['x'])
    y_raw = int(data['y'])
    
    # Validate point using existing function
    pos = nearest_free_black(env, y_raw, x_raw)
    
    if pos is None:
        return jsonify({
            'success': False,
            'error': '❌ Ese punto cae sobre un muro. Selecciona un área libre (negra).'
        }), 400
    
    y, x = pos
    agent = session['agent']
    stage = get_current_stage_name()
    
    # Store point
    session['points'][str(agent)][stage] = [int(y), int(x)]
    
    # Advance to next stage/agent
    if session['stage'] < 2:
        session['stage'] += 1
    else:
        session['stage'] = 0
        if session['agent'] < N_AGENTS - 1:
            session['agent'] += 1
    
    session.modified = True
    
    return jsonify({
        'success': True,
        'point': [int(y), int(x)],
        'agent': agent,
        'stage': stage,
        'next_agent': session['agent'],
        'next_stage': session['stage'],
        'next_stage_name': get_current_stage_name()
    })

@app.route('/api/optimize', methods=['POST'])
def optimize():
    """Run NSGA-II optimization"""
    init_session()
    
    # Validate all points are selected
    points_ok = all(
        len(session['points'][str(a)]) == 3 
        for a in range(N_AGENTS)
    )
    
    if not points_ok:
        return jsonify({
            'success': False,
            'error': '🚫 Faltan puntos para algún agente. Completa todos los puntos primero.'
        }), 400
    
    # Extract points
    starts = [tuple(session['points'][str(a)]['start']) for a in range(N_AGENTS)]
    picks = [tuple(session['points'][str(a)]['pickup']) for a in range(N_AGENTS)]
    drops = [tuple(session['points'][str(a)]['drop']) for a in range(N_AGENTS)]
    
    # Run optimization using existing function
    try:
        result = run_ga_multi_streamlit(
            env,
            starts=starts,
            picks=picks,
            drops=drops,
            pop_size=120,
            ngen=70,
            cxpb=0.6,
            mutpb=0.3,
            seed=0
        )
        
        # Store routes in session
        session['routes'] = result['paths']
        session.modified = True
        
        return jsonify({
            'success': True,
            'routes': result['paths']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error durante la optimización: {str(e)}'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset application state"""
    session.clear()
    init_session()
    return jsonify({'success': True})

# =============================================================
# HTML TEMPLATE (Embedded)
# =============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 Hospital MAOPP - Planificador de Rutas</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
            font-weight: 400;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
            animation: fadeInDown 0.8s ease;
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 10px;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 20px;
            margin-bottom: 20px;
        }

        .card {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.15);
            animation: fadeInUp 0.8s ease;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(255, 255, 255, 0.2);
        }

        .card h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .map-container {
            position: relative;
            background: #1a1a1a;
            border-radius: 20px;
            overflow: hidden;
            cursor: crosshair;
            box-shadow: 0 8px 30px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.1);
            transition: box-shadow 0.3s ease;
        }
        
        .map-container:hover {
            box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 0 2px rgba(79, 172, 254, 0.3);
        }

        .map-container canvas {
            display: block;
            width: 100%;
            height: auto;
            image-rendering: -moz-crisp-edges;
            image-rendering: -webkit-crisp-edges;
            image-rendering: pixelated;
            image-rendering: crisp-edges;
        }
        
        .map-legend {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 12px;
            font-size: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            min-width: 150px;
        }
        
        .legend-title {
            font-weight: 700;
            margin-bottom: 10px;
            color: #4facfe;
            font-size: 0.9rem;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 6px 0;
        }
        
        .legend-symbol {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 2px solid #fff;
            flex-shrink: 0;
        }
        
        .legend-symbol.square {
            border-radius: 3px;
        }
        
        .legend-symbol.cross::before {
            content: '✕';
            color: #fff;
            font-size: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
        }

        .status-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .status-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid rgba(255, 255, 255, 0.15);
            transition: all 0.3s ease;
        }
        
        .status-card:hover {
            background: rgba(255, 255, 255, 0.15);
            border-color: rgba(79, 172, 254, 0.4);
            transform: translateX(2px);
        }

        .status-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 5px;
        }

        .status-value {
            font-size: 1.8rem;
            font-weight: 700;
        }

        .stage-indicator {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }

        .stage-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transition: all 0.3s ease;
        }

        .stage-dot.active {
            background: #00ff88;
            box-shadow: 0 0 16px #00ff88;
            transform: scale(1.3);
        }

        .btn {
            padding: 16px 32px;
            font-size: 1rem;
            font-weight: 600;
            border: none;
            border-radius: 14px;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            text-transform: uppercase;
            letter-spacing: 0.8px;
            position: relative;
            overflow: hidden;
        }
        
        .btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.2);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .btn:hover::before {
            width: 300px;
            height: 300px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }

        .btn-danger {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }

        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(245, 87, 108, 0.4);
        }

        .btn-success {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
        }

        .btn-success:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
        }

        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }

        .button-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .points-list {
            max-height: 400px;
            overflow-y: auto;
        }

        .agent-points {
            margin-bottom: 15px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            border-left: 4px solid;
        }

        .point-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 0.95rem;
        }

        .alert {
            padding: 15px 20px;
            border-radius: 12px;
            margin: 10px 0;
            animation: slideIn 0.3s ease;
        }

        .alert-error {
            background: rgba(245, 87, 108, 0.3);
            border: 1px solid rgba(245, 87, 108, 0.6);
        }

        .alert-success {
            background: rgba(0, 255, 136, 0.3);
            border: 1px solid rgba(0, 255, 136, 0.6);
        }

        #animation-container {
            margin-top: 20px;
        }
        
        .metrics-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .metric-label {
            font-size: 0.85rem;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 700;
        }
        
        .metric-value.warning {
            color: #fbbf24;
        }
        
        .metric-value.danger {
            color: #ef4444;
        }

        .loader {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #fff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 Hospital MAOPP</h1>
            <p>Planificador Inteligente de Rutas Multi-Agente con NSGA-II</p>
        </div>

        <div class="main-grid">
            <!-- Map Section -->
            <div class="card">
                <h2>🗺️ Mapa Hospitalario</h2>
                <div class="map-container" id="map-container">
                    <canvas id="map-canvas"></canvas>
                    
                    <!-- Legend -->
                    <div class="map-legend">
                        <div class="legend-title">Leyenda</div>
                        <div class="legend-item">
                            <div class="legend-symbol" style="background: #ff4444;"></div>
                            <span>● Inicio</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-symbol square" style="background: #ff4444;"></div>
                            <span>■ Recogida</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-symbol cross" style="background: transparent;"></div>
                            <span>✕ Entrega</span>
                        </div>
                    </div>
                </div>
                <div id="pending-point" style="margin-top: 15px; font-size: 0.9rem; opacity: 0.8;"></div>
                <div id="alert-container"></div>
            </div>

            <!-- Control Panel -->
            <div class="status-panel">
                <div class="card">
                    <h2>📋 Estado del Sistema</h2>
                    <div class="status-card">
                        <div class="status-label">Agente Actual</div>
                        <div class="status-value" id="current-agent">1 / 4</div>
                    </div>
                    <div class="status-card">
                        <div class="status-label">Etapa</div>
                        <div class="status-value" id="current-stage">START</div>
                        <div class="stage-indicator">
                            <div class="stage-dot active" data-stage="0"></div>
                            <div class="stage-dot" data-stage="1"></div>
                            <div class="stage-dot" data-stage="2"></div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>🎯 Puntos Seleccionados</h2>
                    <div class="points-list" id="points-list"></div>
                </div>

                <div class="card">
                    <h2>⚙️ Acciones</h2>
                    <div class="button-group">
                        <button class="btn btn-success" id="btn-optimize" disabled>
                            🚀 Optimizar Rutas (NSGA-II)
                        </button>
                        <button class="btn btn-danger" id="btn-reset">
                            🔄 Reset
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Animation Section -->
        <div class="card" id="animation-container" style="display: none;">
            <h2>🎬 Rutas Optimizadas</h2>
            
            <!-- Metrics Panel -->
            <div class="metrics-panel" id="metrics-panel">
                <div class="metric-card">
                    <div class="metric-label">⚠️ Colisiones Detectadas</div>
                    <div class="metric-value" id="metric-collisions">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">📏 Distancia Mínima entre Agentes</div>
                    <div class="metric-value" id="metric-min-dist">--</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">📊 Distancia Total Recorrida</div>
                    <div class="metric-value" id="metric-total-dist">--</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">⏱️ Tiempo Total (frames)</div>
                    <div class="metric-value" id="metric-time">--</div>
                </div>
            </div>
            
            <div id="plotly-animation"></div>
        </div>
    </div>

    <script>
        const colors = ['#ff4444', '#00ffff', '#ffff00', '#00ff00'];
        const stageNames = ['start', 'pickup', 'drop'];
        const stageMarkers = { 'start': '●', 'pickup': '■', 'drop': '✕' };
        
        let state = null;
        let canvas = null;
        let ctx = null;
        let mapImage = null;
        let pendingClick = null;

        // Initialize
        async function init() {
            canvas = document.getElementById('map-canvas');
            ctx = canvas.getContext('2d');
            
            await loadState();
            await loadMap();
            
            canvas.addEventListener('click', handleMapClick);
            document.getElementById('btn-optimize').addEventListener('click', handleOptimize);
            document.getElementById('btn-reset').addEventListener('click', handleReset);
            
            // Auto-refresh state every second
            setInterval(loadState, 1000);
        }

        async function loadState() {
            const response = await fetch('/api/state');
            state = await response.json();
            updateUI();
        }

        async function loadMap() {
            return new Promise((resolve) => {
                mapImage = new Image();
                mapImage.onload = () => {
                    // Scale up for better quality
                    const scale = 4;
                    canvas.width = mapImage.width * scale;
                    canvas.height = mapImage.height * scale;
                    drawMap();
                    resolve();
                };
                mapImage.src = '/api/map?t=' + Date.now();
            });
        }

        function drawMap() {
            if (!mapImage || !ctx) return;
            
            const scale = 4;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Disable smoothing for crisp pixels
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(mapImage, 0, 0, canvas.width, canvas.height);
            ctx.imageSmoothingEnabled = true;
            
            // Draw all selected points
            if (state && state.points) {
                for (let a = 0; a < 4; a++) {
                    const agentPoints = state.points[a.toString()];
                    if (!agentPoints) continue;
                    
                    Object.entries(agentPoints).forEach(([stageName, point]) => {
                        const [y, x] = point;
                        drawPoint(x * scale, y * scale, colors[a], stageName, a);
                    });
                }
            }
            
            // Draw pending click
            if (pendingClick) {
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(pendingClick.x, pendingClick.y, 12, 0, 2 * Math.PI);
                ctx.stroke();
            }
        }

        function drawPoint(x, y, color, stageName, agentNum) {
            const markers = { 'start': '●', 'pickup': '■', 'drop': '✕' };
            const sizes = { 'start': 4, 'pickup': 5, 'drop': 6 };
            
            ctx.fillStyle = color;
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 1.5;
            
            // Draw marker
            ctx.beginPath();
            ctx.arc(x, y, sizes[stageName], 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Draw label
            ctx.font = 'bold 10px Inter';
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 2.5;
            const label = `${agentNum + 1}-${stageName[0].toUpperCase()}`;
            ctx.strokeText(label, x + 10, y + 4);
            ctx.fillText(label, x + 10, y + 4);
        }

        function handleMapClick(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const scale = 4;
            const x = Math.floor((e.clientX - rect.left) * scaleX / scale);
            const y = Math.floor((e.clientY - rect.top) * scaleY / scale);
            
            pendingClick = { x, y };
            drawMap();
            
            fetch('/api/select-point', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x, y })
            })
            .then(async response => {
                const data = await response.json();
                if (!response.ok) {
                    showAlert(data.error, 'error');
                } else {
                    showAlert(`✓ Punto confirmado para Agente ${data.agent + 1} - ${data.stage.toUpperCase()}`, 'success');
                    await loadState();
                    drawMap();
                }
                pendingClick = null;
            })
            .catch(error => {
                showAlert('Error al seleccionar punto', 'error');
                pendingClick = null;
            });
        }

        function updateUI() {
            if (!state) return;
            
            // Update status
            document.getElementById('current-agent').textContent = `${state.agent + 1} / 4`;
            document.getElementById('current-stage').textContent = state.stage_name.toUpperCase();
            
            // Update stage indicators
            document.querySelectorAll('.stage-dot').forEach((dot, idx) => {
                dot.classList.toggle('active', idx === state.stage);
            });
            
            // Update points list
            const pointsList = document.getElementById('points-list');
            pointsList.innerHTML = '';
            
            for (let a = 0; a < 4; a++) {
                const agentPoints = state.points[a.toString()];
                const numPoints = Object.keys(agentPoints || {}).length;
                
                const agentDiv = document.createElement('div');
                agentDiv.className = 'agent-points';
                agentDiv.style.borderLeftColor = colors[a];
                
                let html = `<div style="font-weight: 700; margin-bottom: 10px;">Agente ${a + 1} (${numPoints}/3)</div>`;
                
                if (agentPoints) {
                    Object.entries(agentPoints).forEach(([stage, point]) => {
                        const [y, x] = point;
                        html += `
                            <div class="point-item">
                                <span>${stageMarkers[stage]} ${stage.toUpperCase()}</span>
                                <span style="font-family: monospace;">(${y}, ${x})</span>
                            </div>
                        `;
                    });
                }
                
                agentDiv.innerHTML = html;
                pointsList.appendChild(agentDiv);
            }
            
            // Enable optimize button if all points selected
            const allPointsSelected = Object.values(state.points).every(
                agentPoints => Object.keys(agentPoints).length === 3
            );
            document.getElementById('btn-optimize').disabled = !allPointsSelected;
            
            drawMap();
        }

        async function handleOptimize() {
            const btn = document.getElementById('btn-optimize');
            btn.disabled = true;
            btn.innerHTML = '<div class="loader"></div> Optimizando...';
            
            try {
                const response = await fetch('/api/optimize', { method: 'POST' });
                const data = await response.json();
                
                if (!response.ok) {
                    showAlert(data.error, 'error');
                } else {
                    showAlert('✓ Optimización completada!', 'success');
                    await loadState();
                    displayRoutes(data.routes);
                }
            } catch (error) {
                showAlert('Error durante la optimización', 'error');
            }
            
            btn.disabled = false;
            btn.innerHTML = '🚀 Optimizar Rutas (NSGA-II)';
        }

        async function handleReset() {
            if (!confirm('¿Seguro que quieres resetear todos los puntos?')) return;
            
            await fetch('/api/reset', { method: 'POST' });
            await loadState();
            document.getElementById('animation-container').style.display = 'none';
            showAlert('✓ Sistema reseteado', 'success');
        }

        function showAlert(message, type) {
            const container = document.getElementById('alert-container');
            const alert = document.createElement('div');
            alert.className = `alert alert-${type}`;
            alert.textContent = message;
            container.innerHTML = '';
            container.appendChild(alert);
            
            setTimeout(() => {
                alert.remove();
            }, 3000);
        }

        function displayRoutes(routes) {
            const container = document.getElementById('animation-container');
            container.style.display = 'block';
            container.scrollIntoView({ behavior: 'smooth' });
            
            const maxLen = Math.max(...routes.map(r => r.length));
            
            // Calculate metrics for all frames
            const metrics = [];
            for (let t = 0; t < maxLen; t++) {
                const positions = routes.map((route, i) => {
                    const idx = Math.min(t, route.length - 1);
                    return route[idx];
                });
                
                // Calculate collisions (distance < 6 pixels)
                let collisions = 0;
                let minDist = Infinity;
                for (let i = 0; i < positions.length; i++) {
                    for (let j = i + 1; j < positions.length; j++) {
                        const dy = positions[i][0] - positions[j][0];
                        const dx = positions[i][1] - positions[j][1];
                        const dist = Math.sqrt(dx * dx + dy * dy);
                        minDist = Math.min(minDist, dist);
                        if (dist < 6) collisions++;
                    }
                }
                
                // Calculate total distance
                let totalDist = 0;
                routes.forEach(route => {
                    const idx = Math.min(t, route.length - 1);
                    for (let i = 1; i <= idx; i++) {
                        const dy = route[i][0] - route[i-1][0];
                        const dx = route[i][1] - route[i-1][1];
                        totalDist += Math.sqrt(dx * dx + dy * dy);
                    }
                });
                
                metrics.push({
                    collisions,
                    minDist: minDist === Infinity ? 0 : minDist,
                    totalDist,
                    time: t
                });
            }
            
            // Update initial metrics
            updateMetrics(metrics[0]);
            
            // Prepare traces for each agent (scaled 4x to match map)
            const scale = 4;
            const traces = routes.map((route, i) => ({
                x: [route[0][1] * scale],
                y: [(state.map_size.height - route[0][0]) * scale],
                mode: 'lines+markers',
                line: { color: colors[i], width: 3 },
                marker: { size: 6 },
                name: `Agente ${i + 1}`
            }));
            
            // Add start/pickup/drop markers (scaled 4x)
            routes.forEach((route, a) => {
                const agentPoints = state.points[a.toString()];
                Object.entries(agentPoints).forEach(([stage, point]) => {
                    const [y, x] = point;
                    traces.push({
                        x: [x * scale],
                        y: [(state.map_size.height - y) * scale],
                        mode: 'markers+text',
                        marker: {
                            size: 12,
                            color: colors[a],
                            symbol: stage === 'start' ? 'circle' : (stage === 'pickup' ? 'square' : 'x'),
                            line: { width: 2, color: '#fff' }
                        },
                        text: [`${a + 1}-${stage[0].toUpperCase()}`],
                        textposition: 'top center',
                        textfont: { size: 10, color: '#fff', family: 'Inter, sans-serif' },
                        showlegend: false
                    });
                });
            });
            
            // Create frames (scaled 4x)
            const frames = [];
            for (let t = 0; t < maxLen; t++) {
                const frameData = routes.map((route, i) => {
                    const points = route.slice(0, t + 1);
                    return {
                        x: points.map(p => p[1] * scale),
                        y: points.map(p => (state.map_size.height - p[0]) * scale)
                    };
                });
                frames.push({ 
                    data: frameData, 
                    name: t.toString(),
                    layout: {
                        title: { 
                            text: `Evolución de Rutas con NSGA-II - Frame ${t}/${maxLen}`,
                            font: { 
                                size: 22,
                                color: '#4facfe',
                                family: 'Inter, sans-serif',
                                weight: 700
                            }
                        }
                    }
                });
            }
            
            
            const layout = {
                width: 1100,
                height: 700,
                margin: { l: 10, r: 10, t: 80, b: 60 },
                xaxis: { 
                    range: [0, state.map_size.width * 4],  // 4x scaled
                    showgrid: false,
                    zeroline: false,
                    showticklabels: false
                },
                yaxis: { 
                    range: [0, state.map_size.height * 4],  // 4x scaled
                    scaleanchor: 'x',
                    showgrid: false,
                    zeroline: false,
                    showticklabels: false
                },
                // Add high-res map as background image
                images: [{
                    source: '/api/map-hires',
                    xref: 'x',
                    yref: 'y',
                    x: 0,
                    y: state.map_size.height * 4,
                    sizex: state.map_size.width * 4,
                    sizey: state.map_size.height * 4,
                    sizing: 'stretch',
                    layer: 'below',
                    opacity: 1
                }],
                plot_bgcolor: '#1a1a1a',
                paper_bgcolor: 'rgba(26, 26, 26, 0.95)',
                font: { 
                    color: '#fff',
                    family: 'Inter, sans-serif',
                    size: 12
                },
                title: {
                    text: 'Evolución de Rutas Optimizadas con NSGA-II',
                    font: { 
                        size: 22,
                        color: '#4facfe',
                        family: 'Inter, sans-serif',
                        weight: 700
                    },
                    y: 0.96,  // Bajado ligeramente para evitar superposición
                    x: 0.5,
                    xanchor: 'center',
                    yanchor: 'top',
                    pad: { t: 10, b: 10 }
                },
                updatemenus: [{
                    type: 'buttons',
                    showactive: true,
                    y: 1.08,  // Ajustado para que no se superponga con el título
                    x: 0.12,  // Movido más a la izquierda
                    xanchor: 'left',
                    buttons: [
                        {
                            label: '▶ Play',
                            method: 'animate',
                            args: [null, {
                                frame: { duration: 100, redraw: true },
                                fromcurrent: true,
                                transition: { duration: 0 },
                                mode: 'immediate'
                            }]
                        },
                        {
                            label: '⏸ Pause',
                            method: 'animate',
                            args: [[null], { 
                                frame: { duration: 0 }, 
                                mode: 'immediate',
                                transition: { duration: 0 }
                            }]
                        }
                    ]
                }],
                sliders: [{
                    active: 0,
                    steps: frames.map((frame, idx) => ({
                        args: [[frame.name], {
                            frame: { duration: 0, redraw: true },
                            mode: 'immediate',
                            transition: { duration: 0 }
                        }],
                        label: idx.toString(),
                        method: 'animate'
                    })),
                    x: 0.1,
                    len: 0.8,
                    y: 0,
                    yanchor: 'top'
                }]
            };
            
            Plotly.newPlot('plotly-animation', traces, layout).then(() => {
                Plotly.addFrames('plotly-animation', frames);
                
                // Listen to animation frames to update metrics
                const plotDiv = document.getElementById('plotly-animation');
                plotDiv.on('plotly_animating', () => {
                    // Get current frame from slider
                    const sliderValue = plotDiv.layout.sliders ? plotDiv.layout.sliders[0].active : 0;
                    if (metrics[sliderValue]) {
                        updateMetrics(metrics[sliderValue]);
                    }
                });
                
                // Also listen to slider changes
                plotDiv.on('plotly_sliderchange', (data) => {
                    const frameIdx = parseInt(data.step.label);
                    if (metrics[frameIdx]) {
                        updateMetrics(metrics[frameIdx]);
                    }
                });
            });
        }
        
        function updateMetrics(metric) {
            document.getElementById('metric-collisions').textContent = metric.collisions;
            document.getElementById('metric-collisions').className = 'metric-value ' + 
                (metric.collisions > 0 ? 'danger' : '');
            
            document.getElementById('metric-min-dist').textContent = metric.minDist.toFixed(2) + ' px';
            document.getElementById('metric-min-dist').className = 'metric-value ' + 
                (metric.minDist < 6 ? 'danger' : (metric.minDist < 10 ? 'warning' : ''));
            
            document.getElementById('metric-total-dist').textContent = metric.totalDist.toFixed(1) + ' px';
            document.getElementById('metric-time').textContent = metric.time;
        }

        // Start the application
        init();
    </script>
</body>
</html>
'''

# =============================================================
# MAIN
# =============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 Hospital MAOPP - Professional Web Frontend")
    print("=" * 60)
    print(f"Map loaded: {MAP_PATH}")
    print(f"Map size: {W} x {H}")
    print(f"Number of agents: {N_AGENTS}")
    print("=" * 60)
    print("\n🚀 Starting Flask server...")
    print("📍 Open your browser at: http://localhost:5000")
    print("\nPress CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)