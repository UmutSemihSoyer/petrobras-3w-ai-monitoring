import os
import glob
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from feature_engineering import extract_features_from_df

app = Flask(__name__)

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

# Global Model Loading
loaded_model_data = None
if os.path.exists(MODEL_PATH):
    try:
        loaded_model_data = joblib.load(MODEL_PATH)
        print(f"Loaded trained AI Model: {loaded_model_data.get('model_name', 'XGBoost')} (Accuracy: {loaded_model_data.get('accuracy', 0)*100:.2f}%)")
    except Exception as e:
        print(f"Error loading model: {e}")

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
        "f1_score": float(loaded_model_data.get("f1_score", 0.0)) if loaded_model_data else 0.0
    }
    
    return jsonify({
        "classes": records,
        "model_info": model_meta
    })

@app.route('/api/file_list/<int:class_id>')
def file_list(class_id):
    folder_path = os.path.join(DATASET_DIR, str(class_id))
    if not os.path.exists(folder_path):
        return jsonify({"files": []})
        
    files = glob.glob(os.path.join(folder_path, "*.parquet"))
    file_names = [os.path.basename(f) for f in sorted(files)[:20]] # top 20 files
    return jsonify({"files": file_names})

@app.route('/api/file_data/<int:class_id>/<file_name>')
def file_data(class_id, file_name):
    file_path = os.path.join(DATASET_DIR, str(class_id), file_name)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404
        
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
            
    # AI Prediction on whole file features
    prediction = None
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
            confidence = float(np.max(probs, axis=1).mean()) if probs is not None else 0.95
            
            pred_info = classes_info.get(most_freq_pred, {})
            prediction = {
                "pred_class_id": most_freq_pred,
                "pred_class_name": pred_info.get("name", "Unknown"),
                "desc": pred_info.get("desc", ""),
                "severity": pred_info.get("severity", "Normal"),
                "color": pred_info.get("color", "#10B981"),
                "confidence": round(confidence * 100, 1),
                "is_match": (most_freq_pred == class_id)
            }
        except Exception as e:
            print(f"Prediction error: {e}")

    return jsonify({
        "file_name": file_name,
        "class_id": class_id,
        "total_points": len(df),
        "timestamps": timestamps,
        "series": series_data,
        "ai_prediction": prediction
    })

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
