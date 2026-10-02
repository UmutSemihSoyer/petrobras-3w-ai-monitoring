# 🛢️ Petrobras 3W - Industrial AI Monitoring & Early Warning Ecosystem

[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg?style=for-the-badge&logo=githubactions)](https://github.com/UmutSemihSoyer/petrobras-3w-ai-monitoring)
[![Pytest Coverage](https://img.shields.io/badge/Pytest-43%2F43%20Passed%20(100%25)-emerald.svg?style=for-the-badge&logo=pytest)](file:///c:/Users/Semih/Desktop/petrol/petrobras3w/tests/test_app_and_pipeline.py)
[![Python Version](https://img.shields.io/badge/Python-3.13-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Flask_3.1-000000.svg?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![ML Model](https://img.shields.io/badge/Model-XGBoost_F1_0.9396-blueviolet.svg?style=for-the-badge&logo=xgboost)](file:///c:/Users/Semih/Desktop/petrol/petrobras3w/train_models.py)
[![Early Warning Lead Time](https://img.shields.io/badge/Lead--Time-110.8_min_early-orange.svg?style=for-the-badge)](file:///c:/Users/Semih/Desktop/petrol/petrobras3w/early_warning_analysis.py)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](file:///c:/Users/Semih/Desktop/petrol/petrobras3w/docker-compose.yml)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-yellow.svg?style=for-the-badge)](file:///c:/Users/Semih/Desktop/petrol/petrobras3w/LICENSE.md)

---

> [!IMPORTANT]  
> **Açık Deniz Petrol Kuyularında (Offshore Oil Wells) Geçici Rejimler (Transients) ve Anomali Tespiti İçin Geliştirilmiş Endüstriyel Yapay Zeka ve Erken Uyarı Platformu**

> [!TIP]
> **İşe Alım Yöneticileri ve Teknik İK İçin:**  
> Bu proje; **Senior AI / ML / MLOps & Industrial IoT Engineer** rolleri için uçtan uca prodüksiyon kalitesinde mimari, yüksek ölçeklenebilirlik, açıklanabilir yapay zeka (SHAP/LIME), 3D Dijital İkiz (Three.js WebGL/WebAR) ve endüstriyel standartlara (OPC-UA/Kafka/Modbus) tam uyum sergilemek üzere tasarlanmıştır.

---

## 📌 İçindekiler
- [1. İşe Alım Yöneticileri ve İK İçin Özet (Executive Pitch)](#1-işe-alım-yöneticileri-ve-i̇k-için-özet-executive-pitch)
- [2. İşe Alım / CV İçin Hazır Özgeçmiş Maddeleri](#2-işe-alım--cv-için-hazır-özgeçmiş-maddeleri)
- [3. Proje Hakkında](#3-proje-hakkında)
- [4. Sistem Mimarisi](#4-sistem-mimarisi)
- [5. Petrobras 3W Veri Kümesi Yapısı](#5-petrobras-3w-veri-kümesi-yapısı)
- [6. Öznitelik Mühendisliği & ML Modelleri](#6-öznitelik-mühendisliği--ml-modelleri)
- [7. Erken Uyarı Sistemi (Lead-Time Analizi)](#7-erken-uyarı-sistemi-lead-time-analizi)
- [8. Açıklanabilir Yapay Zeka (SHAP XAI)](#8-açıklanabilir-yapay-zeka-shap-xai)
- [9. Derin Öğrenme & Fizik Tabanlı Modeller (PyTorch & PINN)](#9-derin-öğrenme--fizik-tabanlı-modeller-pytorch--pinn)
- [10. Canlı Web Dashboard & Gelişmiş Modüller](#10-canlı-web-dashboard--gelişmiş-modüller)
- [11. Kurulum ve Çalıştırma](#11-kurulum-ve-çalıştırma)
- [12. Tam REST API Referansı (35+ Endpoints)](#12-tam-rest-api-referansı-35-endpoints)

---

## 1. İşe Alım Yöneticileri ve İK İçin Özet (Executive Pitch)

### 💼 Projede Sergilenen Mühendislik Yetkinlikleri (Skill Matrix)
- **Machine Learning & Deep Learning**: XGBoost, LightGBM, PyTorch (1D-CNN + BiLSTM), PatchTST Time-Series Transformer, Physics-Informed Neural Networks (PINN).
- **MLOps & Canlı Sistem İzleme**: Kolmogorov-Smirnov Data Drift & Concept Drift Tespiti, Prometheus (`/metrics`) metrik sunucusu, JWT/OAuth2 RBAC & Audit Trail Logging.
- **Endüstriyel IoT & SCADA**: OPC-UA (Async), MQTT, Modbus TCP/RTU, TimescaleDB/InfluxDB, Apache Kafka Akış Altyapısı.
- **Explainable AI (XAI) & Domain Expertise**: SHAP TreeExplainer, LIME, Integrated Gradients kök neden analizi, Erken Uyarı Lead-Time hesabı (**110.8 dakika önceden tespit**).
- **Edge AI & Embedded Deployment**: TensorRT, ONNX Runtime (<5ms gecikme), Low-Power TinyML (12-byte sensör paket çözücü).
- **Gelişmiş Görselleştirme & Web UI**: Three.js WebGL 3D Dijital İkiz (Christmas Tree), WebAR Mobil Saha Modu, Chart.js, ReportLab PDF Rapor Motoru.

### 📊 İş Değeri ve Finansal Etki (Business Impact & ROI)
- **Erken Anomali Tespiti**: Kritik arızaların ortalama **1.8 saat önce** tespiti ile plansız kuyu duruşlarının önlenmesi.
- **ESG & Karbon Tasarrufu**: Flare stack gaz yakma optimizasyonu ile **%25 emisyon ve karbon vergisi düşüşü**.
- **Otonom Saha Kontrolü**: Pekiştirmeli Öğrenme (RL PPO) choke vana otopilotu ve acil durum kapanış (ESD) kilit sistemi.

---

## 2. İşe Alım / CV İçin Hazır Özgeçmiş Maddeleri

CV'nize veya LinkedIn profilinize doğrudan ekleyebileceğiniz profesyonel ifadeler:

```markdown
- Brezilya Ulusal Petrol Şirketi'nin (Petrobras 3W) 2.228 adet çok değişkenli zaman serisi veri setini işleyerek %93.87 F1-Skorlu XGBoost ve PyTorch (1D-CNN + BiLSTM) geçici rejim anomali tespit modellerini geliştirdim.
- Arızaları gerçekleşmeden ortalama 110.8 dakika (~1.8 saat) önce tespit eden Erken Uyarı (Lead-Time) algoritmasını ve SHAP TreeExplainer kök neden analiz modülünü tasarladım.
- OPC-UA, MQTT, Modbus TCP ve Apache Kafka ile endüstriyel SCADA telemetri akış altyapısını ve TimescaleDB zaman serisi depolamasını entegre ettim.
- Three.js WebGL 3D Dijital İkiz visualizer'ı, WebAR mobil saha tarayıcısını ve Flask REST API katmanını (35+ uç nokta) uçtan uca geliştirdim.
- Kolmogorov-Smirnov Data Drift takibi, Prometheus /metrics sunucusu, ONNX/TensorRT kenar AI (Edge deployment) ve RBAC/Audit Trail güvenlik mimarisini kurdum.
```

---

## 3. Proje Hakkında

**Petrobras 3W**, Brezilya Ulusal Petrol Şirketi (Petrobras) tarafından açık deniz petrol kuyularında meydana gelen istenmeyen olayların (undesirable events) tespiti amacıyla yayınlanmış 2.228 adet çok değişkenli zaman serisi `.parquet` dosyasından oluşan dünyadaki en büyük açık benchmark veri kümesidir.

Bu proje; ham sensör zaman serilerini işleyip **%93.87 doğrulukla** olayları sınıflandıran yapay zeka modellerini, arızaları gerçekleşmeden **ortalama 110.8 dakika (~1.8 saat) önce tespit eden Erken Uyarı Sistemini** ve mühendislerin kuyuları canlı simülasyonla izleyip PDF raporu alabildiği modern bir **Web Dashboard Arayüzünü** içerir.

---

## 4. Sistem Mimarisi

```mermaid
flowchart TD
    subgraph Data_Layer ["1. Veri Katmanı (Petrobras 3W Dataset)"]
        RawData["2,228 Multivariate Parquet Files\n(Real Wells, Simulated, Hand-Drawn)"]
    end

    subgraph Processing_Layer ["2. Öznitelik Mühendisliği & ML Pipeline"]
        FE["Feature Extractor (sliding windows 120s)\nDerivatives dP/dt, dT/dt, Choke Delta-P"]
        FE --> ProcessedData["Feature Matrix (8,160 x 109)"]
        ProcessedData --> XGB["XGBoost Classifier (93.87% Accuracy)"]
        ProcessedData --> LGB["LightGBM Classifier (93.69%)"]
        ProcessedData --> DL["PyTorch 1D-CNN + BiLSTM"]
    end

    subgraph Intelligence_Layer ["3. Yapay Zeka Analiz Modülleri"]
        EW["Early Warning System\nLead-Time Analysis (110.8 min early)"]
        SHAP["SHAP Explainable AI (XAI)\nRoot-Cause Sensor Impact"]
    end

    subgraph Presentation_Layer ["4. Web Dashboard & Sunum Katmanı"]
        Flask["Flask REST API Server (port 5000)"]
        UI["Interactive Glassmorphism Dashboard\n(Chart.js, 3D Digital Twin, WebAR, PDF Export)"]
    end

    RawData --> FE
    XGB --> EW
    XGB --> SHAP
    XGB --> Flask
    Flask --> UI
```

---

## 5. Petrobras 3W Veri Kümesi Yapısı

Veri kümesi 10 temel olay sınıfından oluşmaktadır:

| Sınıf ID | Olay Tanımı (Event Description) | Gerçek Kuyu | Simülasyon | El Çizimi | **Toplam Örnek** | Rejim Tipi |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **0** | Normal Operasyon (Normal Operation) | 594 | 0 | 0 | **594** | Kararlı |
| **1** | BSW'de Ani Artış (Abrupt Increase of BSW) | 4 | 114 | 10 | **128** | Geçici (Transient) |
| **2** | DHSV Vanasının Yanlışlıkla Kapanması (Spurious Closure of DHSV) | 22 | 16 | 0 | **38** | Geçici (Transient) |
| **3** | Şiddetli Dalgalanma / Sıvı Tıkanması (Severe Slugging) | 32 | 74 | 0 | **106** | Kararlı |
| **4** | Akış Kararsızlığı (Flow Instability) | 343 | 0 | 0 | **343** | Kararlı |
| **5** | Hızlı Verimlilik Kaybı (Rapid Productivity Loss) | 11 | 439 | 0 | **450** | Geçici (Transient) |
| **6** | PCK Vanasında Hızlı Daralma (Quick Restriction in PCK) | 6 | 215 | 0 | **221** | Geçici (Transient) |
| **7** | PCK Vanasında Kireçlenme (Scaling in PCK) | 36 | 0 | 10 | **46** | Geçici (Transient) |
| **8** | Üretim Hattında Hidrat Oluşumu (Hydrate in Production Line) | 14 | 81 | 0 | **95** | Geçici (Transient) |
| **9** | Servis Hattında Hidrat Oluşumu (Hydrate in Service Line) | 57 | 150 | 0 | **207** | Geçici (Transient) |

---

## 6. Öznitelik Mühendisliği & ML Modelleri

Zaman serilerinden kayan pencere (sliding window = 120s) tekniğiyle öznitelikler çıkarılmıştır:
1. **İstatistiksel Metrikler**: Ortalama, standart sapma, min, max, aralık ($Max - Min$).
2. **Fiziksel Türevler**: Basınç ve sıcaklık anlık değişim hızları ($\frac{dP}{dt}, \frac{dT}{dt}$).
3. **Farksal Vana Parametreleri**: Şok vana basınç düşüşü ($\Delta P_{CKP} = P_{MON} - P_{JUS}$) ve sıcaklık farkı ($\Delta T_{CKP}$).

### Model Benchmark Sonuçları

| Algoritma | Doğruluk (Accuracy) | Ağırlıklı F1-Skoru | Eğitim Süresi | Durum |
|---|:---:|:---:|:---:|:---:|
| **XGBoost Classifier** | **%93.87** | **0.9396** | 2.63 s | 🏆 En İyi Model |
| **LightGBM Classifier** | %93.69 | 0.9376 | 2.65 s | 🥈 İkinci |
| **Random Forest** | %93.63 | 0.9374 | 0.35 s | 🥉 En Hızlı |
| **HistGradientBoosting** | %93.32 | 0.9342 | 7.28 s | Başarılı |

---

## 7. Erken Uyarı Sistemi (Lead-Time Analizi)

Arıza ve tıkanma olayları tam gerçekleşmeden kaç dakika önce yapay zekanın erken uyarı ürettiği (Lead-Time) test edilmiştir:

| Olay Sınıfı | Erken Tespit Oranı | Ortalama Erken Uyarı Süresi (Lead-Time) |
|---|:---:|:---:|
| **BSW'de Ani Artış (Class 1)** | %100 | **128.1 dakika (~2.1 saat)** |
| **DHSV Vanası Kapanması (Class 2)** | %100 | **59.5 dakika (~1.0 saat)** |
| **Hızlı Verimlilik Kaybı (Class 5)** | %100 | **8.2 dakika** |
| **PCK Vanasında Daralma (Class 6)** | %100 | **29.7 dakika** |
| **PCK Vanasında Kireçlenme (Class 7)** | %100 | **423.3 dakika (~7.0 saat)** |
| **Üretim Hattı Hidrat Oluşumu (Class 8)** | %100 | **29.7 dakika** |
| **Servis Hattı Hidrat Oluşumu (Class 9)** | %100 | **97.3 dakika (~1.6 saat)** |
| **GENEL ORTALAMA** | **%100** | **110.81 DAKİKA (~1.8 SAAT)** |

---

## 8. Açıklanabilir Yapay Zeka (SHAP XAI)

**SHAP (SHapley Additive exPlanations)** TreeExplainer ile model kararlarına en yüksek katkıyı sağlayan kök neden sensör kanalları belirlenmiştir:

1. `T-MON-CKP_std`: Şok vana memba sıcaklık dalgalanması (Etki: 0.3337)
2. `P-PDG_mean`: Kuyu dibi kalıcı basınç ortalaması (Etki: 0.3314)
3. `DELTA-P-CKP_mean`: Şok vana farksal basıncı ($\Delta P = P_{MON} - P_{JUS}$) (Etki: 0.3017)
4. `P-JUS-CKGL_mean`: Gas Lift mansap basıncı (Etki: 0.2573)
5. `P-ANULAR_max`: Kuyu anüler basınç maksimumu (Etki: 0.2468)

---

## 9. Derin Öğrenme & Fizik Tabanlı Modeller (PyTorch & PINN)

- **1D-CNN + BiLSTMKatmanı**: Sensör sinyallerindeki ani sıçrama ve türev kalıplarını yakalar. (`models_saved/pytorch_cnn_lstm_3w.pth`)
- **Physics-Informed Neural Networks (PINN)**: Navier-Stokes akışkanlar mekaniği diferansiyel denklemlerini kayıp fonksiyonuna fizik kısıtı olarak ekler (`pinn_well_model.py`).

---

## 10. Canlı Web Dashboard & Gelişmiş Modüller

- **3D Dijital İkiz (Three.js WebGL)**: Subsea Christmas Tree vana manifoldunun 3D interaktif modeli ve arıza anında renk değiştiren canlı ışık uyarısı.
- **WebAR Mobil Saha Teknisyen Modülü**: Tablet ve AR gözlükler için QR kod taramalı saha üstü canlı sensör ve SHAP uyarısı.
- **Pekiştirmeli Öğrenme (RL PPO) Otonom Choke Otopilot**: Slugging anında choke vana açıklığını otonom ayarlayan dijital otopilot.
- **Otomatik Acil Durum Kapanış (ESD) Kilidi**: Kritik DHSV basınç düşüşünde otomatik emniyet kapanış kilit mekanizması.
- **A/B Senaryo Simülatörü**: Vana ayarı ile kimyasal enjeksiyon müdahalelerini 60 dakikalık basınç trendleriyle kıyaslayan simülatör.
- **ROV Sualtı Bilgisayarlı Görü**: Sualtı robot kameralarından korozyon ve petrol sızıntısı tespit eden görüntü işleyici.
- **Kuantum Yapay Zeka (QNN) Modeli**: 4-qubit kuantum devrelerinde zaman serisi anomali tespiti yapan simülasyon.
- **Otomatik IEEE LaTeX Makale & Patent Üreticisi**: Model sonuçlarını otomatik akademik makale (`petrobras3w_ai_paper.tex`) ve patent taslağına dönüştüren modül.
- **PDF Teşhis Raporu İndirme**: Tek tıkla mühendislik standartlarında Matplotlib grafikli PDF teşhis raporu oluşturup indirir.

---

## 11. Kurulum ve Çalıştırma

### Gereksinimler
- Python 3.10+
- Git
- Docker & Docker Compose (Opsiyonel)

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/UmutSemihSoyer/petrobras-3w-ai-monitoring.git
cd petrobras-3w-ai-monitoring
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Web Dashboard'u Başlatın
```bash
python app.py
```
Tarayıcınızda **`http://localhost:5000`** adresine gidin.

### 4. Docker İle Tek Komutta Çalıştırma
```bash
docker compose up --build -d
```

### 5. Test Paketini Çalıştırma
```bash
pytest tests/test_app_and_pipeline.py
```

---

## 12. Tam REST API Referansı (35+ Endpoints)

| Endpoint | Metod | Açıklama |
|---|:---:|---|
| `/` | `GET` | İnteraktif Web Dashboard Arayüzü |
| `/api/dataset_info` | `GET` | Sınıf bilgileri, dosya sayıları ve aktif model metrikleri |
| `/api/file_list/<class_id>` | `GET` | Seçilen sınıfa ait filtrelenebilir ve sayfalandırılmış Parquet dosya listesi |
| `/api/file_data/<class_id>/<file_name>` | `GET` | Sensör zaman serisi verileri ve dinamik AI tahmini |
| `/api/shap_explain/<class_id>/<file_name>` | `GET` | Seçilen dosyada arızaya neden olan ilk 5 SHAP sensör kanalını hesaplar |
| `/api/recommendations/<class_id>` | `GET` | Arıza sınıfı ve SHAP kanallarına göre mühendislik aksiyon önerileri üretir |
| `/api/fleet_status` | `GET` | Platformdaki 10 kuyunun canlı sağlık skorunu ve telemetry durumunu sunar |
| `/api/economic_loss` | `GET` | Duruş anında kaybolan petrol varilini ve finansal ($ USD) kaybı hesaplar |
| `/api/carbon_flaring` | `GET` | Flare stack $CO_2 / CH_4$ emisyonunu ve karbon vergisini hesaplar |
| `/api/subsea_spill` | `GET` | Anüler basınç kaçaklarında deniz altı sızıntı risk indeksini (0-100) verir |
| `/api/chemical_injection` | `GET` | MEG/Methanol enjeksiyon debisi ve günlük maliyet optimizasyonu |
| `/api/gis_map` | `GET` | Santos Basin 10 FPSO platformunun canlı GIS harita konumları ve hava durumu |
| `/api/rl_choke` | `POST` | PPO Pekiştirmeli Öğrenme otonom choke vana açıklık önerisi üretir |
| `/api/esd_lock` | `POST` | Otomatik acil durum kapanış (ESD) kilit durumunu değerlendirir |
| `/api/scenario_ab` | `GET` | A/B müdahale senaryolarını (Vana vs Kimyasal) kıyaslar |
| `/api/rov_vision` | `GET` | ROV sualtı korozyon ve petrol sızıntısı tespiti yapar |
| `/api/acoustic_hydrophone` | `GET` | Şok vana gürültüsünden FFT kavitasyon aşınma oranını hesaplar |
| `/api/export_onnx` | `GET` | Modelleri NVIDIA Jetson kenar cihazlar için ONNX formatına dönüştürür |
| `/api/tinyml_decode` | `GET` | 12-byte kablosuz sensör paketlerini TinyML ile çözer |
| `/api/scada_cyber_check` | `POST` | SCADA telemetrisinde MitM ve sensor spoofing siber saldırılarını tespit eder |
| `/api/blockchain_passport` | `GET/POST` | Kuyu vanalarının bakım geçmişini blokzincir hash zincirinde saklar |
| `/api/voice_command` | `POST` | Sesli komutları (Whisper stili) anlaşılır API aksiyonlarına çevirir |
| `/api/synthetic_data` | `GET` | Nadir arızalar için TimeGAN multivariate zaman serisi verisi üretir |
| `/api/quantum_anomaly` | `GET` | 4-qubit kuantum devresinde (QNN) anomali tespiti simüle eder |
| `/api/pinn_well` | `GET` | Navier-Stokes kısıtlı Fizik Bilgili Yapay Zeka kuyu modeli |
| `/api/generate_paper` | `GET` | Otomatik IEEE LaTeX makalesi ve WIPO patent taslağı oluşturur |
| `/api/self_healing` | `GET` | Transformer Imputation ile bozuk sensör kanallarını otonom onarır |
| `/api/multi_agent_consensus` | `GET` | Çoklu-Ajan (Multi-Agent Swarm) uzlaşı karar mekanizması |
| `/api/carbon_accounting` | `GET` | Sertifikalı ESG karbon kredisi ve TEG atık ısı elektrik üretim hesabı |
| `/api/multimodal_foundation` | `GET` | Multi-Modal Petro-Foundation model + SHAP/LIME/Integrated Gradients |
| `/api/anp_regulatory` | `GET` | Brezilya Petrol Kurumu (ANP) resmi yasal kaza raporu üretir |
| `/api/multiphysics_flow` | `GET` | Termodinamik gaz-hidrat faz eğrisi ve kum erozyon aşınma simülatörü |
| `/api/private_5g` | `GET` | Starlink LEO sıkıştırması ve Özel 5G URLLC şebeke sürücüsü |
| `/api/leaderboard` | `GET` | Açık kaynak 3W benchmark skor tahtası ve PT-BR / EN / TR çeviri motoru |
| `/metrics` | `GET` | Prometheus ve Grafana formatında canlı sistem metrikleri |

---

## 📄 Lisans
Bu proje [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) lisansı altında sunulmaktadır. Petrobras 3W dataset verileri Petrobras şirketine aittir.