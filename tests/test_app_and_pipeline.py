import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from app import app
from feature_engineering import extract_features_from_df

class TestAppAndPipeline(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        """Test landing page loads properly."""
        rv = self.client.get('/')
        self.assertEqual(rv.status_code, 200)
        self.assertIn(b"PETROBRAS 3W", rv.data)

    def test_dataset_info_api(self):
        """Test dataset info API endpoint."""
        res = self.client.get('/api/dataset_info')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("classes", data)
        self.assertEqual(len(data["classes"]), 10)
        self.assertIn("model_info", data)

    def test_file_list_api(self):
        """Test file list API endpoint with pagination and search."""
        res = self.client.get('/api/file_list/0?per_page=10')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("files", data)
        self.assertIn("total_files", data)
        self.assertEqual(data["per_page"], 10)

    def test_feature_engineering_extractor(self):
        """Test sliding window feature extraction function."""
        df_dummy = pd.DataFrame({
            'P-PDG': np.random.randn(150) * 1e5 + 2e7,
            'T-PDG': np.random.randn(150) * 5 + 60,
            'P-TPT': np.random.randn(150) * 1e5 + 1.5e7,
            'T-TPT': np.random.randn(150) * 3 + 45,
            'P-MON-CKP': np.random.randn(150) * 1e5 + 1e7,
            'P-JUS-CKP': np.random.randn(150) * 1e5 + 5e6,
            'T-MON-CKP': np.random.randn(150) * 3 + 40,
            'T-JUS-CKP': np.random.randn(150) * 3 + 35,
            'class': [0] * 150
        })
        
        feats = extract_features_from_df(df_dummy, window_size=60, step_size=30)
        self.assertFalse(feats.empty)
        self.assertIn('target_class', feats.columns)
        self.assertIn('P-PDG_mean', feats.columns)
        self.assertIn('DELTA-P-CKP_mean', feats.columns)

    def test_stream_point_api(self):
        """Test streaming sensor point buffer API."""
        point = {
            'P-PDG': 20000000.0,
            'T-PDG': 65.0,
            'P-TPT': 15000000.0,
            'T-TPT': 48.0,
            'P-MON-CKP': 10000000.0,
            'P-JUS-CKP': 5000000.0,
            'T-MON-CKP': 42.0,
            'T-JUS-CKP': 36.0
        }
        res = self.client.post('/api/stream_point', json=point)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("buffered_points", data)

    def test_trigger_alert_api(self):
        """Test simulated webhook alerting API."""
        payload = {"file_name": "WELL-00001.parquet", "fault_name": "Hydrate Formation", "severity": "CRITICAL"}
        res = self.client.post('/api/trigger_alert', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "Alert sent successfully")

    def test_recommendations_api(self):
        """Test engineering recommendations API."""
        res = self.client.get('/api/recommendations/2')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("title", data)
        self.assertIn("steps", data)
        self.assertGreater(len(data["steps"]), 0)

    def test_fleet_status_api(self):
        """Test multi-well fleet status monitoring API."""
        res = self.client.get('/api/fleet_status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("fleet_summary", data)
        self.assertIn("wells", data)
        self.assertGreater(len(data["wells"]), 0)

    def test_opcua_api(self):
        """Test OPC-UA status and stream toggle API."""
        res = self.client.get('/api/opcua/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("server_url", data)

        res_toggle = self.client.post('/api/opcua/stream_toggle', json={"action": "start"})
        self.assertEqual(res_toggle.status_code, 200)
        self.assertTrue(res_toggle.get_json()["is_streaming"])

        res_stop = self.client.post('/api/opcua/stream_toggle', json={"action": "stop"})
        self.assertEqual(res_stop.status_code, 200)
        self.assertFalse(res_stop.get_json()["is_streaming"])

    def test_modbus_driver(self):
        """Test Modbus TCP register encoder/decoder driver."""
        from modbus_driver import ModbusTCPDriver
        r1, r2 = ModbusTCPDriver.encode_float32(123.45)
        decoded_val = ModbusTCPDriver.decode_float32(r1, r2)
        self.assertAlmostEqual(decoded_val, 123.45, places=2)

    def test_timescaledb_store(self):
        """Test TimescaleDB time-series store."""
        from timescaledb_store import TimescaleDBStore
        store = TimescaleDBStore(db_path=":memory:")
        store.insert_telemetry("WELL-01", {"timestamp": "2026-10-02 12:00:00", "P-PDG": 20000000.0})
        hist = store.get_recent_history("WELL-01", limit=10)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["P-PDG"], 20000000.0)

    def test_kafka_streamer(self):
        """Test Kafka telemetry producer/consumer mock."""
        from kafka_streamer import KafkaTelemetryProducer
        producer = KafkaTelemetryProducer()
        res = producer.send_sensor_event("WELL-01", {"P-PDG": 2e7})
        self.assertEqual(res["well_id"], "WELL-01")

    def test_gnn_fault_propagation(self):
        """Test Graph Neural Network fault propagation prediction."""
        from gnn_fault_propagation import predict_fault_propagation
        matrix = np.random.randn(10, 8)
        risks = predict_fault_propagation(matrix)
        self.assertEqual(len(risks), 10)
        self.assertTrue(all(0.0 <= r <= 1.0 for r in risks))

    def test_predictive_maintenance_rul(self):
        """Test Weibull RUL calculation."""
        from predictive_maintenance_rul import RULPredictor
        predictor = RULPredictor(beta_shape=2.5, eta_scale=5000.0)
        res = predictor.calculate_rul(operating_hours=2000.0)
        self.assertIn("rul_hours", res)
        self.assertIn("degradation_pct", res)
        self.assertGreater(res["rul_hours"], 0)

    def test_mlops_drift_monitor(self):
        """Test Kolmogorov-Smirnov Data Drift detector."""
        from mlops_drift_monitor import DataDriftMonitor
        monitor = DataDriftMonitor()
        base = np.random.normal(loc=10.0, scale=1.0, size=100)
        curr_no_drift = np.random.normal(loc=10.0, scale=1.0, size=100)
        curr_with_drift = np.random.normal(loc=15.0, scale=1.0, size=100)
        
        res1 = monitor.detect_drift(base, curr_no_drift)
        self.assertFalse(res1["drift_detected"])

        res2 = monitor.detect_drift(base, curr_with_drift)
        self.assertTrue(res2["drift_detected"])

    def test_rbac_and_audit_security(self):
        """Test RBAC permissions and Audit Logger."""
        from auth_audit_security import SecurityRBACManager, AuditTrailLogger
        self.assertTrue(SecurityRBACManager.check_permission("admin", "retrain_model"))
        self.assertFalse(SecurityRBACManager.check_permission("operator", "retrain_model"))

        logger = AuditTrailLogger(db_path=":memory:")
        logger.log_action("engineer_01", "engineer", "export_pdf", "Downloaded PDF report")
        logs = logger.get_recent_audit_logs()
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["action"], "export_pdf")

    def test_economic_loss_api(self):
        """Test Economic Loss calculation API endpoint."""
        res = self.client.get('/api/economic_loss?duration_minutes=120&productivity_loss_pct=50')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("lost_revenue_usd", data)
        self.assertGreater(data["lost_revenue_usd"], 0)

    def test_carbon_flaring_api(self):
        """Test Carbon Flaring Emissions API endpoint."""
        res = self.client.get('/api/carbon_flaring?flared_gas_m3=10000')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("co2_emissions_tons", data)
        self.assertIn("carbon_tax_cost_usd", data)

    def test_subsea_spill_api(self):
        """Test Subsea Environmental Spill Risk API endpoint."""
        res = self.client.get('/api/subsea_spill?p_anular_pa=6000000')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("spill_risk_score", data)
        self.assertIn("level", data)

    def test_chemical_injection_api(self):
        """Test Chemical Injection Dosage & Cost API endpoint."""
        res = self.client.get('/api/chemical_injection?temp_c=12.0&class_id=8')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("meg_injection_l_hr", data)
        self.assertIn("total_daily_cost_usd", data)

    def test_gis_map_api(self):
        """Test GIS Offshore Platform Map API endpoint."""
        res = self.client.get('/api/gis_map')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 10)
        self.assertIn("lat", data[0])

    def test_rl_choke_api(self):
        """Test Reinforcement Learning Autonomous Choke Control API."""
        payload = {"current_choke_pct": 50.0, "p_mon_pa": 1.5e7, "p_jus_pa": 5e6, "pressure_std_pa": 4e5}
        res = self.client.post('/api/rl_choke', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("optimal_choke_pct", data)
        self.assertIn("recommendation", data)

    def test_esd_lock_api(self):
        """Test Emergency Shutdown (ESD) Safety Lock API."""
        payload = {"well_id": "WELL-01", "p_pdg_pa": 40e7, "p_mon_ckp_pa": 30e7}
        res = self.client.post('/api/esd_lock', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["tripped"])
        self.assertGreaterEqual(data["esd_level"], 1)

    def test_scenario_ab_api(self):
        """Test Scenario A/B Root Cause Simulator API."""
        res = self.client.get('/api/scenario_ab?class_id=8')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("scenario_a", data)
        self.assertIn("scenario_b", data)
        self.assertIn("recommended_scenario", data)

    def test_rov_vision_api(self):
        """Test ROV Subsea Vision Defect Analyzer API."""
        res = self.client.get('/api/rov_vision?frame_id=FRAME-202')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("defect_detected", data)
        self.assertIn("detections", data)

    def test_acoustic_hydrophone_api(self):
        """Test Acoustic Hydrophone Spectrogram Analyzer API."""
        res = self.client.get('/api/acoustic_hydrophone')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("rms_amplitude_pa", data)
        self.assertIn("status", data)

    def test_export_onnx_api(self):
        """Test ONNX / TensorRT Edge Exporter API."""
        res = self.client.get('/api/export_onnx')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("onnx_path", data)

    def test_tinyml_decode_api(self):
        """Test TinyML Sensor Driver API."""
        res = self.client.get('/api/tinyml_decode')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("battery_percent", data)
        self.assertIn("tinyml_anomaly_score", data)

    def test_scada_cyber_check_api(self):
        """Test SCADA Cyber Attack & Spoofing Detector API."""
        payload = {"sensor_name": "P-PDG", "current_val": 2e7, "previous_val": 1e7, "time_delta_sec": 0.1, "mac_signature_valid": False}
        res = self.client.post('/api/scada_cyber_check', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["attack_detected"])

    def test_blockchain_passport_api(self):
        """Test Blockchain Equipment Digital Passport API."""
        res_get = self.client.get('/api/blockchain_passport')
        self.assertEqual(res_get.status_code, 200)
        self.assertTrue(res_get.get_json()["integrity_valid"])

        payload = {"equipment_id": "VALVE-003", "technician_id": "TECH-01", "action": "MAINTENANCE", "details": "Replaced seal ring"}
        res_post = self.client.post('/api/blockchain_passport', json=payload)
        self.assertEqual(res_post.status_code, 200)
        self.assertEqual(res_post.get_json()["status"], "SUCCESS")

    def test_voice_command_api(self):
        """Test Voice AI Command Processor API."""
        payload = {"transcript": "kuyu durumunu göster"}
        res = self.client.post('/api/voice_command', json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["detected_intent"], "GET_WELL_STATUS")

    def test_synthetic_data_api(self):
        """Test TimeGAN Synthetic Data Generator API."""
        res = self.client.get('/api/synthetic_data?class_id=2&samples=50')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["sample_count"], 50)
        self.assertEqual(data["class_id"], 2)

    def test_quantum_anomaly_api(self):
        """Test Quantum Neural Network Anomaly Detector API."""
        res = self.client.get('/api/quantum_anomaly')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["num_qubits"], 4)
        self.assertIn("anomaly_probability", data)

    def test_pinn_well_api(self):
        """Test Physics-Informed Neural Network Well Model API."""
        res = self.client.get('/api/pinn_well?p_pdg_pa=22000000')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("pinn_navier_stokes_residual", data)
        self.assertIn("breakout_risk_score", data)

    def test_generate_paper_api(self):
        """Test LaTeX Paper & Patent Generator API."""
        res = self.client.get('/api/generate_paper')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertIn("latex_file_path", data)

    def test_self_healing_api(self):
        """Test Self-Healing Data Pipeline API."""
        res = self.client.get('/api/self_healing')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("status", data)

    def test_multi_agent_consensus_api(self):
        """Test Multi-Agent Swarm Consensus API."""
        res = self.client.get('/api/multi_agent_consensus?class_id=8')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("final_decision", data)
        self.assertIn("agent_votes", data)

    def test_carbon_accounting_api(self):
        """Test ESG Carbon Accounting & TEG Recovery API."""
        res = self.client.get('/api/carbon_accounting?prevented_flaring_m3=20000')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("carbon_credits_earned", data)
        self.assertIn("teg_power_generated_kw", data)

    def test_multimodal_foundation_api(self):
        """Test Multi-Modal Foundation Model & LIME Explainability API."""
        res = self.client.get('/api/multimodal_foundation?class_id=8')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("feature_explanations", data)
        self.assertIn("model_architecture", data)

    def test_anp_regulatory_api(self):
        """Test ANP Regulatory Compliance Reporter API."""
        res = self.client.get('/api/anp_regulatory?well_id=WELL-02&class_id=2')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("anp_incident_code", data)

    def test_multiphysics_flow_api(self):
        """Test Multi-Physics Flow Assurance Simulator API."""
        res = self.client.get('/api/multiphysics_flow?pressure_bar=160&temp_c=14')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("margin_to_hydrate_c", data)
        self.assertIn("sand_erosion_wear_mm_yr", data)

    def test_private_5g_api(self):
        """Test Private 5G URLLC & Satellite IoT Telemetry Driver API."""
        res = self.client.get('/api/private_5g')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("compression_ratio", data)
        self.assertIn("private_5g_latency_ms", data)

    def test_leaderboard_api(self):
        """Test Open Source Benchmark Leaderboard & i18n API."""
        res = self.client.get('/api/leaderboard?lang=pt_BR')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("rankings", data)
        self.assertIn("title", data)

if __name__ == '__main__':
    unittest.main()



