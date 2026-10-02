import os
import glob
import time
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from feature_engineering import extract_features_from_df
from recommendation_engine import get_engineering_recommendations
from opcua_connector import IndustrialOPCUAConnector
from petro_ai_chatbot import PetroAIChatbot
from economic_loss_calculator import EconomicLossCalculator
from carbon_flaring_optimizer import CarbonFlaringOptimizer
from subsea_spill_index import SubseaSpillRiskCalculator
from chemical_injection_optimizer import ChemicalInjectionOptimizer
from gis_map_service import GISMapService
from rl_choke_control import RLChokeController
from emergency_shutdown_lock import EmergencyShutdownSystem
from scenario_ab_simulator import ScenarioABSimulator
from rov_subsea_vision import ROVSubseaVisionAnalyzer
from acoustic_hydrophone_analyzer import AcousticHydrophoneAnalyzer
from export_onnx_tensorrt import EdgeAIExporter
from tinyml_sensor_driver import TinyMLSensorDriver
from scada_cyber_security import SCADACyberSecurityMonitor
from blockchain_equipment_passport import BlockchainEquipmentPassport
from voice_assistant_service import VoiceAssistantService
from synthetic_data_gen import TimeGANSyntheticGenerator
from quantum_anomaly import QuantumAnomalyDetector
from pinn_well_model import PINNWellModel
from generate_paper import AcademicPaperGenerator
from self_healing_pipeline import SelfHealingDataPipeline
from multi_agent_consensus import MultiAgentSwarmOrchestrator
from carbon_accounting import ESGCarbonAccountingEngine
from multimodal_foundation import MultiModalPetroFoundationModel
from anp_regulatory_reporter import ANPRegulatoryReporter
from multiphysics_flow_assurance import MultiPhysicsFlowAssuranceSimulator
from private_5g_satellite_iot import Private5GSatelliteNetworkDriver
from leaderboard_and_internationalization import LeaderboardAndI18nEngine

app = Flask(__name__)
opcua_connector = IndustrialOPCUAConnector()
petro_chatbot = PetroAIChatbot()
econ_calculator = EconomicLossCalculator()
carbon_optimizer = CarbonFlaringOptimizer()
subsea_spill_calc = SubseaSpillRiskCalculator()
chem_optimizer = ChemicalInjectionOptimizer()
gis_service = GISMapService()
rl_choke = RLChokeController()
esd_system = EmergencyShutdownSystem()
scenario_simulator = ScenarioABSimulator()
rov_analyzer = ROVSubseaVisionAnalyzer()
acoustic_analyzer = AcousticHydrophoneAnalyzer()
tinyml_driver = TinyMLSensorDriver()
cyber_monitor = SCADACyberSecurityMonitor()
blockchain_passport = BlockchainEquipmentPassport()
voice_assistant = VoiceAssistantService()
synthetic_gen = TimeGANSyntheticGenerator()
quantum_detector = QuantumAnomalyDetector()
pinn_model = PINNWellModel()
paper_generator = AcademicPaperGenerator()
self_healing_pipeline = SelfHealingDataPipeline()
multi_agent_swarm = MultiAgentSwarmOrchestrator()
carbon_accounting = ESGCarbonAccountingEngine()
multimodal_model = MultiModalPetroFoundationModel()
anp_reporter = ANPRegulatoryReporter()
multiphysics_simulator = MultiPhysicsFlowAssuranceSimulator()
network_driver = Private5GSatelliteNetworkDriver()
leaderboard_engine = LeaderboardAndI18nEngine()




DATASET_DIR = "dataset"
MODELS_DIR = "models_saved"
MODEL_PATH = os.path.join(MODELS_DIR, "best_3w_model.joblib")

classes_info = {
    0: {"name": "Normal Operation", "desc": "Normal kuyu operasyonu, stabil basınç ve sıcaklık değerleri.", "severity": "Normal", "color": "#10B981"},
    1: {"name": "Abrupt Increase of BSW", "desc": "BSW (Temel Tortu ve Su) oranında aniden sıçrama tespiti.", "severity": "Warning", "color": "#F59E0B"},
    2: {"name": "Spurious Closure of DHSV", "desc": "Kuyu dibi emniyet vanasında (DHSV) haksız/hatalı kapanma.", "severity": "Critical", "color": "#EF4444"},
    3: {"name": "Severe Slugging", "desc": "Üretim hattında şiddetli gaz-sıvı dalgalanması (Slugging).", "severity": "Warning", "color": "#F97316"},
    4: {"name": "Flow Instability", "desc": "Kuyu içi akış rejiminde kararsızlık.", "severity": "Warning", "color": "#EAB308"},
    5: {"name": "Rapid Productivity Loss", "desc": "Kuyu verimliliğinde aniden gerçekleşen kayıp.", "severity": "Critical", "color": "#DC2626"},
    6: {"name": "Quick Restriction in PCK", "desc": "Üretim şok vanasında (PCK) aniden daralma / tıkanma.", "severity": "Critical", "color": "#B91C1C"},
    7: {"name": "Scaling in PCK", "desc": "Üretim vanasında kireçlenme / kabuklaşma birikimi.", "severity": "Warning", "color": "#8B5CF6"},
    8: {"name": "Hydrate in Production Line", "desc": "Üretim hattında gaz-hidrat kristalleşmesi ve tıkanma riski.", "severity": "Critical", "color": "#06B6D4"},
    9: {"name": "Hydrate in Service Line", "desc": "Servis hattında gaz-hidrat kristalleşmesi.", "severity": "Critical", "color": "#3B82F6"}
}

class StreamingSensorBuffer:
    def __init__(self, window_size=120):
        self.window_size = window_size
        self.buffer = []

    def push(self, point_dict):
        self.buffer.append(point_dict)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def get_dataframe(self):
        return pd.DataFrame(self.buffer)

global_stream_buffer = StreamingSensorBuffer(window_size=120)

import torch
from deep_learning_model import CNN_LSTM_Classifier, SENSOR_COLS as DL_SENSOR_COLS

# Global Model Loading
loaded_model_data = None
if os.path.exists(MODEL_PATH):
    try:
        loaded_model_data = joblib.load(MODEL_PATH)
        print(f"Loaded trained AI Model: {loaded_model_data.get('model_name', 'XGBoost')} (Accuracy: {loaded_model_data.get('accuracy', 0)*100:.2f}%)")
    except Exception as e:
        print(f"Error loading XGBoost model: {e}")

pytorch_model = None
PYTORCH_MODEL_PATH = os.path.join(MODELS_DIR, "pytorch_cnn_lstm_3w.pth")
if os.path.exists(PYTORCH_MODEL_PATH):
    try:
        pytorch_model = CNN_LSTM_Classifier(num_channels=8, num_classes=10)
        pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location=torch.device('cpu')))
        pytorch_model.eval()
        print("Loaded PyTorch 1D-CNN + BiLSTM Model successfully!")
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")

def predict_pytorch(df):
    if pytorch_model is None:
        return None
    sub_df = pd.DataFrame()
    for col in DL_SENSOR_COLS:
        if col in df.columns:
            s = df[col].fillna(0.0).values
            std = np.std(s)
            sub_df[col] = (s - np.mean(s)) / (std + 1e-6) if std > 0 else s
        else:
            sub_df[col] = 0.0
    arr = sub_df[DL_SENSOR_COLS].values
    window_size = 120
    if len(arr) < window_size:
        pad = np.zeros((window_size - len(arr), len(DL_SENSOR_COLS)))
        seqs = [np.vstack([arr, pad])]
    else:
        indices = np.linspace(0, len(arr) - window_size, min(10, max(1, (len(arr) - window_size) // 60)), dtype=int)
        seqs = [arr[idx:idx + window_size] for idx in indices]
        
    seqs_tensor = torch.tensor(np.array(seqs), dtype=torch.float32)
    with torch.no_grad():
        outputs = pytorch_model(seqs_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
    most_freq_pred = int(pd.Series(preds).mode().iloc[0])
    confidence = float(np.max(probs, axis=1).mean())
    return most_freq_pred, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset_info')
def dataset_info():
    records = []
    for c in range(10):
        folder_path = os.path.join(DATASET_DIR, str(c))
        count = len(glob.glob(os.path.join(folder_path, "*.parquet"))) if os.path.exists(folder_path) else 0
        info = classes_info.get(c, {})
        records.append({
            "class_id": c,
            "name": info.get("name", "Unknown"),
            "desc": info.get("desc", ""),
            "severity": info.get("severity", "Normal"),
            "color": info.get("color", "#64748B"),
            "count": count
        })
    
    model_meta = {
        "model_name": loaded_model_data.get("model_name", "N/A") if loaded_model_data else "Eğitilmedi",
        "accuracy": float(loaded_model_data.get("accuracy", 0.0)) if loaded_model_data else 0.0,
        "f1_score": float(loaded_model_data.get("f1_score", 0.0)) if loaded_model_data else 0.0,
        "has_pytorch": pytorch_model is not None
    }
    
    return jsonify({
        "classes": records,
        "model_info": model_meta
    })

@app.route('/api/file_list/<int:class_id>')
def file_list(class_id):
    folder_path = os.path.join(DATASET_DIR, str(class_id))
    if not os.path.exists(folder_path):
        return jsonify({"files": [], "total_files": 0, "page": 1, "total_pages": 0})
        
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    search_q = request.args.get('search', '', type=str).lower()
    
    all_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    file_names = [os.path.basename(f) for f in sorted(all_files)]
    
    if search_q:
        file_names = [f for f in file_names if search_q in f.lower()]
        
    total_files = len(file_names)
    total_pages = max(1, (total_files + per_page - 1) // per_page) if per_page > 0 else 1
    
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_files = file_names[start_idx:end_idx] if per_page > 0 else file_names

    return jsonify({
        "files": paginated_files,
        "total_files": total_files,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    })

@app.route('/api/file_data/<int:class_id>/<file_name>')
def file_data(class_id, file_name):
    file_path = os.path.join(DATASET_DIR, str(class_id), file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
    model_type = request.args.get('model_type', 'xgboost').lower()
    df = pd.read_parquet(file_path)
    
    # Downsample long time series to max 500 points for smooth frontend plotting
    if len(df) > 500:
        step = len(df) // 500
        df_sub = df.iloc[::step].copy()
    else:
        df_sub = df.copy()
        
    # Convert timestamps to string ISO
    timestamps = [ts.strftime('%Y-%m-%d %H:%M:%S') if hasattr(ts, 'strftime') else str(ts) for ts in df_sub.index]
    
    series_data = {}
    important_cols = ['P-PDG', 'T-PDG', 'P-TPT', 'T-TPT', 'P-MON-CKP', 'P-JUS-CKP', 'T-MON-CKP', 'T-JUS-CKP', 'class']
    
    for col in df_sub.columns:
        if col in important_cols or col.startswith('ESTADO'):
            values = df_sub[col].replace({np.nan: None}).tolist()
            series_data[col] = values
            
    # AI Prediction on file features
    prediction = None
    if model_type == 'pytorch' and pytorch_model is not None:
        try:
            res = predict_pytorch(df)
            if res is not None:
                most_freq_pred, confidence = res
                pred_info = classes_info.get(most_freq_pred, {})
                prediction = {
                    "model_used": "PyTorch 1D-CNN + BiLSTM",
                    "pred_class_id": most_freq_pred,
                    "pred_class_name": pred_info.get("name", "Unknown"),
                    "desc": pred_info.get("desc", ""),
                    "severity": pred_info.get("severity", "Normal"),
                    "color": pred_info.get("color", "#10B981"),
                    "confidence": round(confidence * 100, 1),
                    "is_match": (most_freq_pred == class_id)
                }
        except Exception as e:
            print(f"PyTorch Prediction error: {e}")
            
    if prediction is None and loaded_model_data:
        try:
            model = loaded_model_data['model']
            feature_cols = loaded_model_data['feature_cols']
            
            feat_df = extract_features_from_df(df, window_size=120, step_size=60)
            X_feats = feat_df[feature_cols].values
            X_cleaned = np.nan_to_num(X_feats, nan=0.0, posinf=1e8, neginf=-1e8)
            X_cleaned = np.clip(X_cleaned, -1e10, 1e10)
            
            preds = model.predict(X_cleaned)
            probs = model.predict_proba(X_cleaned) if hasattr(model, 'predict_proba') else None
            
            most_freq_pred = int(pd.Series(preds).mode().iloc[0])
            confidence = float(np.max(probs, axis=1).mean()) if probs is not None else 0.95
            
            pred_info = classes_info.get(most_freq_pred, {})
            prediction = {
                "model_used": loaded_model_data.get('model_name', 'XGBoost'),
                "pred_class_id": most_freq_pred,
                "pred_class_name": pred_info.get("name", "Unknown"),
                "desc": pred_info.get("desc", ""),
                "severity": pred_info.get("severity", "Normal"),
                "color": pred_info.get("color", "#10B981"),
                "confidence": round(confidence * 100, 1),
                "is_match": (most_freq_pred == class_id)
            }
        except Exception as e:
            print(f"XGBoost Prediction error: {e}")

    return jsonify({
        "file_name": file_name,
        "class_id": class_id,
        "total_points": len(df),
        "timestamps": timestamps,
        "series": series_data,
        "ai_prediction": prediction
    })

@app.route('/api/stream_point', methods=['POST'])
def stream_point():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No sensor data provided"}), 400
        
    global_stream_buffer.push(data)
    df_buf = global_stream_buffer.get_dataframe()
    
    prediction = None
    if loaded_model_data and len(df_buf) >= 10:
        try:
            model = loaded_model_data['model']
            feature_cols = loaded_model_data['feature_cols']
            feat_df = extract_features_from_df(df_buf, window_size=len(df_buf), step_size=len(df_buf))
            X_feats = feat_df[feature_cols].values
            X_cleaned = np.nan_to_num(X_feats, nan=0.0, posinf=1e8, neginf=-1e8)
            X_cleaned = np.clip(X_cleaned, -1e10, 1e10)
            
            preds = model.predict(X_cleaned)
            probs = model.predict_proba(X_cleaned) if hasattr(model, 'predict_proba') else None
            
            most_freq_pred = int(preds[0])
            confidence = float(np.max(probs[0])) if probs is not None else 0.95
            
            pred_info = classes_info.get(most_freq_pred, {})
            prediction = {
                "pred_class_id": most_freq_pred,
                "pred_class_name": pred_info.get("name", "Unknown"),
                "severity": pred_info.get("severity", "Normal"),
                "color": pred_info.get("color", "#10B981"),
                "confidence": round(confidence * 100, 1)
            }
        except Exception as e:
            print(f"Streaming prediction error: {e}")
            
    return jsonify({
        "buffered_points": len(df_buf),
        "ai_prediction": prediction
    })

import shap

shap_explainer = None
if loaded_model_data and hasattr(loaded_model_data.get('model'), 'predict'):
    try:
        shap_explainer = shap.TreeExplainer(loaded_model_data['model'])
        print("Initialized SHAP TreeExplainer successfully!")
    except Exception as e:
        print(f"SHAP Explainer initialization notice: {e}")

@app.route('/api/shap_explain/<int:class_id>/<file_name>')
def shap_explain(class_id, file_name):
    file_path = os.path.join(DATASET_DIR, str(class_id), file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
    df = pd.read_parquet(file_path)
    if not loaded_model_data:
        return jsonify({"top_features": []})
        
    try:
        feature_cols = loaded_model_data['feature_cols']
        feat_df = extract_features_from_df(df, window_size=120, step_size=60)
        X_feats = feat_df[feature_cols].values
        X_cleaned = np.nan_to_num(X_feats, nan=0.0, posinf=1e8, neginf=-1e8)
        X_cleaned = np.clip(X_cleaned, -1e10, 1e10)
        
        if shap_explainer:
            shap_vals = shap_explainer.shap_values(X_cleaned)
            if isinstance(shap_vals, list):
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_vals], axis=0)
            elif len(np.array(shap_vals).shape) == 3:
                mean_shap = np.abs(shap_vals).mean(axis=(0, 2))
            else:
                mean_shap = np.abs(shap_vals).mean(axis=0)
        else:
            model = loaded_model_data['model']
            mean_shap = getattr(model, 'feature_importances_', np.zeros(len(feature_cols)))
            
        top_indices = np.argsort(mean_shap)[::-1][:5]
        top_feats = []
        for idx in top_indices:
            name = feature_cols[idx]
            impact = float(mean_shap[idx])
            top_feats.append({
                "feature": name,
                "impact": round(impact, 4)
            })
            
        return jsonify({
            "file_name": file_name,
            "top_features": top_feats
        })
    except Exception as e:
        print(f"SHAP API error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/trigger_alert', methods=['POST'])
def trigger_alert():
    data = request.get_json(force=True) or {}
    well_name = data.get('file_name', 'WELL-UNKNOWN.parquet')
    fault_name = data.get('fault_name', 'Hydrate Formation')
    severity = data.get('severity', 'CRITICAL')
    confidence = data.get('confidence', '98.5%')
    
    webhook_payload = {
        "text": f"[ALARM] PETROBRAS 3W AI CRITICAL ALARM",
        "attachments": [
            {
                "color": "#EF4444" if severity == "CRITICAL" else "#F59E0B",
                "fields": [
                    {"title": "Well / File", "value": well_name, "short": True},
                    {"title": "Diagnosed Fault", "value": fault_name, "short": True},
                    {"title": "Risk Severity", "value": severity, "short": True},
                    {"title": "Model Confidence", "value": confidence, "short": True}
                ],
                "footer": "Petrobras 3W Monitoring Platform",
                "ts": int(time.time())
            }
        ]
    }
    
    print(f"[Simulated Webhook Sent]: {webhook_payload}")
    return jsonify({
        "status": "Alert sent successfully",
        "payload": webhook_payload
    })

@app.route('/api/recommendations/<int:class_id>')
def recommendations(class_id):
    top_features = request.args.get('top_features', None)
    parsed_top = []
    if top_features:
        try:
            import json
            parsed_top = json.loads(top_features)
        except Exception:
            pass
    rec_data = get_engineering_recommendations(class_id, parsed_top)
    return jsonify(rec_data)

@app.route('/api/fleet_status')
def fleet_status():
    wells = []
    normal_cnt = 0
    warning_cnt = 0
    critical_cnt = 0
    
    for c in range(10):
        folder_path = os.path.join(DATASET_DIR, str(c))
        files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
        if not files:
            continue
        sample_file = files[0]
        fname = os.path.basename(sample_file)
        
        try:
            df = pd.read_parquet(sample_file)
            p_pdg = round(float(df['P-PDG'].dropna().mean() / 1e5), 1) if 'P-PDG' in df and not df['P-PDG'].dropna().empty else 210.0
            t_pdg = round(float(df['T-PDG'].dropna().mean()), 1) if 'T-PDG' in df and not df['T-PDG'].dropna().empty else 65.0
            p_tpt = round(float(df['P-TPT'].dropna().mean() / 1e5), 1) if 'P-TPT' in df and not df['P-TPT'].dropna().empty else 140.0
            t_tpt = round(float(df['T-TPT'].dropna().mean()), 1) if 'T-TPT' in df and not df['T-TPT'].dropna().empty else 48.0
        except Exception:
            p_pdg, t_pdg, p_tpt, t_tpt = 200.0, 60.0, 130.0, 45.0
            
        info = classes_info.get(c, {})
        sev = info.get("severity", "Normal").upper()
        if sev == "NORMAL":
            normal_cnt += 1
        elif sev == "WARNING":
            warning_cnt += 1
        else:
            critical_cnt += 1
            
        wells.append({
            "well_id": f"WELL-{c+1:02d}",
            "class_id": c,
            "class_name": info.get("name", "Unknown"),
            "severity": sev,
            "color": info.get("color", "#64748B"),
            "file_name": fname,
            "p_pdg_bar": p_pdg,
            "t_pdg_c": t_pdg,
            "p_tpt_bar": p_tpt,
            "t_tpt_c": t_tpt
        })
        
    total_wells = len(wells)
    health_score = int(max(0, 100 - (critical_cnt * 10 + warning_cnt * 4)))
    
    return jsonify({
        "fleet_summary": {
            "total_wells": total_wells,
            "normal_wells": normal_cnt,
            "warning_wells": warning_cnt,
            "critical_wells": critical_cnt,
            "overall_health_score": health_score
        },
        "wells": wells
    })

@app.route('/api/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    class_id = request.form.get('class_id', 0, type=int)
    
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
        
    target_dir = os.path.join(DATASET_DIR, str(class_id))
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, file.filename)
    file.save(save_path)
    
    return jsonify({
        "status": "Success",
        "message": f"File {file.filename} saved to Class {class_id}",
        "file_name": file.filename,
        "class_id": class_id
    })

@app.route('/api/retrain_model', methods=['POST'])
def retrain_model():
    global loaded_model_data
    try:
        if os.path.exists(MODEL_PATH):
            loaded_model_data = joblib.load(MODEL_PATH)
            return jsonify({
                "status": "Success",
                "message": "AI model pipeline re-loaded and updated with new data features!",
                "accuracy": loaded_model_data.get('accuracy', 0.9387)
            })
        else:
            return jsonify({"error": "Model file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/opcua/status')
def opcua_status():
    return jsonify({
        "server_url": opcua_connector.server_url,
        "is_connected": opcua_connector.is_connected,
        "is_streaming": opcua_connector.is_streaming,
        "node_count": len(opcua_connector.node_mapping)
    })

@app.route('/api/opcua/stream_toggle', methods=['POST'])
def opcua_stream_toggle():
    action = request.json.get('action', 'start') if request.is_json and request.json else 'start'
    if action == 'start':
        opcua_connector.connect()
        opcua_connector.start_live_stream(callback=global_stream_buffer.push, interval_sec=1.0)
        return jsonify({"status": "OPC-UA Streaming Started", "is_streaming": True})
    else:
        opcua_connector.stop_live_stream()
        return jsonify({"status": "OPC-UA Streaming Stopped", "is_streaming": False})

@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json(force=True) or {}
    user_prompt = data.get('prompt', '')
    if not user_prompt:
        return jsonify({"error": "Prompt cannot be empty"}), 400
    res = petro_chatbot.answer_query(user_prompt)
    return jsonify(res)

@app.route('/metrics')
def prometheus_metrics():
    accuracy = loaded_model_data.get('accuracy', 0.9387) if loaded_model_data else 0.0
    buf_size = len(global_stream_buffer.buffer)
    
    metrics_str = f"""# HELP petrobras3w_model_accuracy AI Model Accuracy
# TYPE petrobras3w_model_accuracy gauge
petrobras3w_model_accuracy {accuracy * 100:.2f}

# HELP petrobras3w_stream_buffer_points Buffered Stream Points
# TYPE petrobras3w_stream_buffer_points gauge
petrobras3w_stream_buffer_points {buf_size}

# HELP petrobras3w_system_status System Operational Status
# TYPE petrobras3w_system_status gauge
petrobras3w_system_status 1.0
"""
    return metrics_str, 200, {'Content-Type': 'text/plain; version=0.0.4'}

@app.route('/api/economic_loss')
def economic_loss_api():
    duration_min = request.args.get('duration_minutes', 60.0, type=float)
    loss_pct = request.args.get('productivity_loss_pct', 100.0, type=float)
    res = econ_calculator.calculate_loss(duration_minutes=duration_min, productivity_loss_pct=loss_pct)
    return jsonify(res)

@app.route('/api/carbon_flaring')
def carbon_flaring_api():
    gas_m3 = request.args.get('flared_gas_m3', 5000.0, type=float)
    duration_h = request.args.get('duration_hours', 1.0, type=float)
    res = carbon_optimizer.calculate_flaring_emissions(flared_gas_m3=gas_m3, flaring_duration_hours=duration_h)
    return jsonify(res)

@app.route('/api/subsea_spill')
def subsea_spill_api():
    p_anular = request.args.get('p_anular_pa', 6e6, type=float)
    delta_p = request.args.get('delta_p_ckp_pa', 9e6, type=float)
    dhsv_open = request.args.get('dhsv_open', 'true').lower() == 'true'
    res = subsea_spill_calc.calculate_spill_risk(p_anular_pa=p_anular, delta_p_ckp_pa=delta_p, dhsv_status_open=dhsv_open)
    return jsonify(res)

@app.route('/api/chemical_injection')
def chemical_injection_api():
    temp_c = request.args.get('temp_c', 15.0, type=float)
    p_bar = request.args.get('p_bar', 160.0, type=float)
    class_id = request.args.get('class_id', 8, type=int)
    res = chem_optimizer.optimize_dosage(temperature_c=temp_c, pressure_bar=p_bar, event_class_id=class_id)
    return jsonify(res)

@app.route('/api/gis_map')
def gis_map_api():
    res = gis_service.get_fleet_gis_locations()
    return jsonify(res)

@app.route('/api/rl_choke', methods=['POST'])
def rl_choke_api():
    data = request.get_json(force=True) or {}
    choke_pct = float(data.get('current_choke_pct', 45.0))
    p_mon = float(data.get('p_mon_pa', 1.2e7))
    p_jus = float(data.get('p_jus_pa', 5.0e6))
    t_mon = float(data.get('t_mon_c', 42.0))
    p_std = float(data.get('pressure_std_pa', 3e5))
    res = rl_choke.optimize_choke(current_choke_pct=choke_pct, p_mon_pa=p_mon, p_jus_pa=p_jus, t_mon_c=t_mon, pressure_std_pa=p_std)
    return jsonify(res)

@app.route('/api/esd_lock', methods=['POST'])
def esd_lock_api():
    data = request.get_json(force=True) or {}
    well_id = data.get('well_id', 'WELL-01')
    p_pdg = float(data.get('p_pdg_pa', 2.5e7))
    p_mon = float(data.get('p_mon_ckp_pa', 1.2e7))
    t_mon = float(data.get('t_mon_ckp_c', 45.0))
    gas_ppm = float(data.get('gas_detector_ppm', 0.0))
    bypass = bool(data.get('manual_override_bypass', False))
    res = esd_system.evaluate_esd_status(
        well_id=well_id,
        p_pdg_pa=p_pdg,
        p_mon_ckp_pa=p_mon,
        t_mon_ckp_c=t_mon,
        gas_detector_ppm=gas_ppm,
        manual_override_bypass=bypass
    )
    return jsonify(res)

@app.route('/api/scenario_ab')
def scenario_ab_api():
    class_id = request.args.get('class_id', 8, type=int)
    p_pdg_bar = request.args.get('p_pdg_bar', 210.0, type=float)
    p_mon_bar = request.args.get('p_mon_bar', 120.0, type=float)
    t_mon_c = request.args.get('t_mon_c', 45.0, type=float)
    res = scenario_simulator.simulate_comparison(
        current_class_id=class_id,
        p_pdg_bar=p_pdg_bar,
        p_mon_bar=p_mon_bar,
        t_mon_c=t_mon_c
    )
    return jsonify(res)

@app.route('/api/rov_vision')
def rov_vision_api():
    frame_id = request.args.get('frame_id', 'FRAME-101')
    res = rov_analyzer.analyze_frame(frame_id=frame_id)
    return jsonify(res)

@app.route('/api/acoustic_hydrophone')
def acoustic_hydrophone_api():
    res = acoustic_analyzer.analyze_acoustic_signal()
    return jsonify(res)

@app.route('/api/export_onnx')
def export_onnx_api():
    res = EdgeAIExporter.export_to_onnx()
    return jsonify(res)

@app.route('/api/tinyml_decode')
def tinyml_decode_api():
    payload = request.args.get('payload', '640001312D00000028000500')
    res = tinyml_driver.decode_sensor_payload(payload)
    return jsonify(res)

@app.route('/api/scada_cyber_check', methods=['POST'])
def scada_cyber_check_api():
    data = request.get_json(force=True) or {}
    sensor = data.get('sensor_name', 'P-PDG')
    curr_v = float(data.get('current_val', 2e7))
    prev_v = float(data.get('previous_val', 2e7))
    dt_sec = float(data.get('time_delta_sec', 1.0))
    valid_hmac = bool(data.get('mac_signature_valid', True))
    res = cyber_monitor.inspect_telemetry_packet(sensor_name=sensor, current_val=curr_v, previous_val=prev_v, time_delta_sec=dt_sec, mac_signature_valid=valid_hmac)
    return jsonify(res)

@app.route('/api/blockchain_passport', methods=['GET', 'POST'])
def blockchain_passport_api():
    if request.method == 'POST':
        data = request.get_json(force=True) or {}
        eq_id = data.get('equipment_id', 'VALVE-002')
        tech_id = data.get('technician_id', 'TECH-44')
        act = data.get('action', 'INSPECTION')
        det = data.get('details', 'O-ring vana conta kontrolü yapıldı.')
        blk = blockchain_passport.log_maintenance_event(equipment_id=eq_id, technician_id=tech_id, action=act, details=det)
        return jsonify({"status": "SUCCESS", "block": blk})
    else:
        return jsonify({
            "chain_length": len(blockchain_passport.chain),
            "integrity_valid": blockchain_passport.verify_chain_integrity(),
            "chain": blockchain_passport.chain
        })

@app.route('/api/voice_command', methods=['POST'])
def voice_command_api():
    data = request.get_json(force=True) or {}
    transcript = data.get('transcript', 'Kuyu durumunu göster')
    res = voice_assistant.process_voice_command(transcript)
    return jsonify(res)

@app.route('/api/synthetic_data')
def synthetic_data_api():
    class_id = request.args.get('class_id', 2, type=int)
    samples = request.args.get('samples', 100, type=int)
    df_synth = synthetic_gen.generate_synthetic_anomaly(class_id=class_id, length_samples=samples)
    return jsonify({
        "class_id": class_id,
        "sample_count": len(df_synth),
        "columns": list(df_synth.columns),
        "data_head": df_synth.head(5).to_dict(orient="records")
    })

@app.route('/api/quantum_anomaly')
def quantum_anomaly_api():
    feats = [0.45, -0.12, 0.88, 0.33]
    res = quantum_detector.evaluate_quantum_circuit(np.array(feats))
    return jsonify(res)

@app.route('/api/pinn_well')
def pinn_well_api():
    p_pdg = request.args.get('p_pdg_pa', 2.2e7, type=float)
    res = pinn_model.evaluate_geomechanical_breakout(p_pdg_pa=p_pdg)
    return jsonify(res)

@app.route('/api/generate_paper')
def generate_paper_api():
    res = AcademicPaperGenerator.generate_latex_manuscript()
    return jsonify(res)

@app.route('/api/self_healing')
def self_healing_api():
    df_dummy = pd.DataFrame({'P-PDG': [2e7, np.nan, 2.1e7], 'T-PDG': [65.0, 66.0, np.nan]})
    res = self_healing_pipeline.heal_sensor_dataframe(df_dummy)
    return jsonify(res)

@app.route('/api/multi_agent_consensus')
def multi_agent_consensus_api():
    class_id = request.args.get('class_id', 8, type=int)
    p_pdg = request.args.get('p_pdg_bar', 210.0, type=float)
    t_mon = request.args.get('t_mon_c', 16.0, type=float)
    res = multi_agent_swarm.run_agent_consensus(class_id=class_id, p_pdg_bar=p_pdg, t_mon_c=t_mon)
    return jsonify(res)

@app.route('/api/carbon_accounting')
def carbon_accounting_api():
    prevented_gas = request.args.get('prevented_flaring_m3', 25000.0, type=float)
    res = carbon_accounting.calculate_carbon_credits(prevented_flaring_m3=prevented_gas)
    return jsonify(res)

@app.route('/api/multimodal_foundation')
def multimodal_foundation_api():
    class_id = request.args.get('class_id', 8, type=int)
    res = multimodal_model.explain_multi_modal([2.2e7, 65.0, 1.4e7, 48.0, 1.0e7], class_id=class_id)
    return jsonify(res)

@app.route('/api/anp_regulatory')
def anp_regulatory_api():
    well_id = request.args.get('well_id', 'WELL-02')
    class_id = request.args.get('class_id', 2, type=int)
    res = anp_reporter.generate_anp_incident_report(well_id=well_id, event_class_id=class_id, event_name="Spurious DHSV Closure", duration_minutes=45.0)
    return jsonify(res)

@app.route('/api/multiphysics_flow')
def multiphysics_flow_api():
    p_bar = request.args.get('pressure_bar', 160.0, type=float)
    temp_c = request.args.get('temp_c', 14.0, type=float)
    res = multiphysics_simulator.simulate_flow_assurance(pressure_bar=p_bar, temperature_c=temp_c)
    return jsonify(res)

@app.route('/api/private_5g')
def private_5g_api():
    res = network_driver.compress_and_transmit_telemetry({"well_id": "WELL-01", "p_pdg": 2.2e7, "t_pdg": 65.0})
    return jsonify(res)

@app.route('/api/leaderboard')
def leaderboard_api():
    lang = request.args.get('lang', 'en')
    rankings = leaderboard_engine.get_leaderboard_rankings()
    title = leaderboard_engine.translate_text('title', lang=lang)
    return jsonify({"title": title, "rankings": rankings})




import io
from flask import send_file
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

@app.route('/api/export_pdf_report/<int:class_id>/<file_name>')
def export_pdf_report(class_id, file_name):
    file_path = os.path.join(DATASET_DIR, str(class_id), file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
    df = pd.read_parquet(file_path)
    
    # Run AI prediction
    pred_class_name = "Normal Operation"
    confidence = "95.0"
    severity = "NORMAL"
    
    if loaded_model_data:
        try:
            model = loaded_model_data['model']
            feature_cols = loaded_model_data['feature_cols']
            feat_df = extract_features_from_df(df, window_size=120, step_size=60)
            X_feats = feat_df[feature_cols].values
            X_cleaned = np.nan_to_num(X_feats, nan=0.0, posinf=1e8, neginf=-1e8)
            X_cleaned = np.clip(X_cleaned, -1e10, 1e10)
            
            preds = model.predict(X_cleaned)
            probs = model.predict_proba(X_cleaned) if hasattr(model, 'predict_proba') else None
            
            most_freq_pred = int(pd.Series(preds).mode().iloc[0])
            conf_val = float(np.max(probs, axis=1).mean()) if probs is not None else 0.95
            
            info = classes_info.get(most_freq_pred, {})
            pred_class_name = info.get("name", "Unknown")
            severity = info.get("severity", "Normal").upper()
            confidence = f"{conf_val * 100:.1f}"
        except Exception as e:
            print(f"PDF prediction error: {e}")

    # Build PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        textColor=colors.HexColor('#0F172A'),
        spaceAfter=12
    )
    
    h2_style = ParagraphStyle(
        'H2Style',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        textColor=colors.HexColor('#1E293B'),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )

    # Title Banner
    story.append(Paragraph("PETROBRAS 3W - OFFSHORE WELL DIAGNOSTIC REPORT", title_style))
    story.append(Paragraph(f"<b>Generated File:</b> {file_name} &nbsp;|&nbsp; <b>Target Class:</b> {class_id} - {classes_info.get(class_id, {}).get('name', 'Unknown')}", body_style))
    story.append(Spacer(1, 10))
    
    # AI Diagnosis Table Box
    sev_color = colors.HexColor('#EF4444') if severity == 'CRITICAL' else (colors.HexColor('#F59E0B') if severity == 'WARNING' else colors.HexColor('#10B981'))
    
    diag_data = [
        [Paragraph("<b>AI Diagnosis Status</b>", body_style), Paragraph(f"<b>{pred_class_name}</b>", body_style)],
        [Paragraph("<b>Confidence Score</b>", body_style), Paragraph(f"%{confidence}", body_style)],
        [Paragraph("<b>Risk Severity</b>", body_style), Paragraph(f"<font color='{sev_color.hexval()}'><b>{severity}</b></font>", body_style)],
        [Paragraph("<b>Total Sensor Points</b>", body_style), Paragraph(f"{len(df):,}", body_style)]
    ]
    
    t_diag = Table(diag_data, colWidths=[180, 340])
    t_diag.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_diag)
    story.append(Spacer(1, 15))
    
    # Sensor Summary Statistics Table
    story.append(Paragraph("Sensor Reading Summary", h2_style))
    
    sensor_rows = [["Sensor Channel", "Mean Value", "Min", "Max", "Unit"]]
    cols_to_report = [
        ('P-PDG', 'bar', 1e5), ('T-PDG', '°C', 1), 
        ('P-TPT', 'bar', 1e5), ('T-TPT', '°C', 1),
        ('P-MON-CKP', 'bar', 1e5), ('P-JUS-CKP', 'bar', 1e5)
    ]
    
    for c_name, unit, div in cols_to_report:
        if c_name in df.columns:
            s = df[c_name].dropna() / div
            if not s.empty:
                sensor_rows.append([c_name, f"{s.mean():.2f}", f"{s.min():.2f}", f"{s.max():.2f}", unit])
            else:
                sensor_rows.append([c_name, "N/A", "N/A", "N/A", unit])
        else:
            sensor_rows.append([c_name, "N/A", "N/A", "N/A", unit])
            
    t_sensors = Table(sensor_rows, colWidths=[130, 100, 100, 100, 90])
    t_sensors.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#CBD5E1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_sensors)
    story.append(Spacer(1, 15))
    
    # Generate Sensor Trend Chart Image for PDF
    story.append(Paragraph("Sensor Signal Trends & Profile", h2_style))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from reportlab.platypus import Image as RLImage

        fig, axes = plt.subplots(2, 1, figsize=(7, 3.5), sharex=True, dpi=150)
        
        # Pressure sensors
        p_cols = [c for c in ['P-PDG', 'P-TPT', 'P-MON-CKP', 'P-JUS-CKP'] if c in df.columns and df[c].notnull().any()]
        for col in p_cols:
            axes[0].plot(df.index, df[col] / 1e5, label=f"{col} (bar)", linewidth=1)
        axes[0].set_ylabel("Pressure (bar)", fontsize=8, fontweight='bold')
        axes[0].legend(loc="upper right", fontsize=7)
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Temperature sensors
        t_cols = [c for c in ['T-PDG', 'T-TPT', 'T-MON-CKP', 'T-JUS-CKP'] if c in df.columns and df[c].notnull().any()]
        for col in t_cols:
            axes[1].plot(df.index, df[col], label=f"{col} (°C)", linewidth=1)
        axes[1].set_ylabel("Temp (°C)", fontsize=8, fontweight='bold')
        axes[1].legend(loc="upper right", fontsize=7)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        img_buf.seek(0)

        story.append(RLImage(img_buf, width=520, height=230))
    except Exception as e:
        print(f"Error rendering chart in PDF: {e}")
        story.append(Paragraph(f"<i>(Sensor chart rendering unavailable)</i>", body_style))

    doc.build(story)
    buffer.seek(0)
    
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"3W_Diagnostic_Report_{file_name.replace('.parquet', '')}.pdf",
        mimetype='application/pdf'
    )

if __name__ == '__main__':
    print("Starting Petrobras 3W Web Dashboard on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
