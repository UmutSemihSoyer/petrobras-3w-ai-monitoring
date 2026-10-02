"""
Petrobras 3W AI Monitoring - Engineering Diagnostic & Action Recommendation Engine

This module processes AI diagnosed fault classes and SHAP root-cause sensor channels
to generate domain-specific petroleum engineering action recommendations.
"""

RECOMMENDATIONS_DATABASE = {
    0: {
        "title": "Normal Operasyon Stabilitesi",
        "action_level": "BİLGİ / RUTİN",
        "color": "#10B981",
        "steps": [
            "Kuyu başı ve kuyu dibi basınç-sıcaklık değerleri nominal sınırlar içerisindedir.",
            "Rutin üretim debisi ve vana pozisyonu kontrollerine devam edilsin.",
            "Herhangi bir müdahale gerekliliği bulunmamaktadır."
        ]
    },
    1: {
        "title": "BSW (Su Oranı) Ani Artış Müdahale Prosedürü",
        "action_level": "YÜKSEK BÖLGESEL RİSK",
        "color": "#F59E0B",
        "steps": [
            "Üretim ayırıcı (seperatör) tank su seviyelerini ve su tahliye vanalarını kontrol edin.",
            "Su emülsiyonunu kırmak için de-emülgatör kimyasal enjeksiyon debisini %20 artırın.",
            "Kuyu dibi su konlanması (water coning) şüphesine karşı üretim debisini %10 oranında kısarak kuyu başı basıncını gözlemleyin."
        ]
    },
    2: {
        "title": "DHSV Emniyet Vanası Yanlışlıkla Kapanma Acil Aksiyonu",
        "action_level": "KRİTİK ACİL MÜDAHALE",
        "color": "#EF4444",
        "steps": [
            "DHSV hidrolik kumanda hattı basıncını (Control Line Pressure) ve aktüatör sızdırmazlığını hemen kontrol edin.",
            "Vana kilit pozisyonunu doğrulamak için kuyu başı emniyet sistemini (ESD - Emergency Shutdown System) denetleyin.",
            "Hidrolik basıncı kademeli olarak yeniden sağlayarak DHSV vanasının tam açık konumda olduğunu sensörlerden teyit edin."
        ]
    },
    3: {
        "title": "Şiddetli Sıvı Birikmesi ve Dalgalanma (Slugging) Engelleme Prosedürü",
        "action_level": "ORTA / YÜKSEK OPERASYONEL RİSK",
        "color": "#F97316",
        "steps": [
            "Üretim şok vanasını (PCK) %10-15 oranında kısarak kuyu üzerinde sabit bir geri basınç (back-pressure) oluşturun.",
            "Slugging döngü frekansını düşürmek için Gas Lift enjeksiyon debisini (QGL) %15 artırın.",
            "Giriş manifoldundaki Slug Catcher sıvı toplama seviyesini ve taşma emniyet vanalarını izleyin."
        ]
    },
    4: {
        "title": "Akış Kararsızlığı Dengeleme Aksiyonları",
        "action_level": "ORTA RİSK",
        "color": "#EAB308",
        "steps": [
            "Choke (PCK) vana açıklığını kademeli olarak sabitleyin ve transient rejim basılmasını bekleyin.",
            "Gas lift enjeksiyon noktasındaki mansap basıncını (P-JUS-CKGL) denetleyin.",
            "Kuyu dibi basınç (P-PDG) ve sıcaklık (T-PDG) trendlerini 30 dakika boyunca kesintisiz izlemeye alın."
        ]
    },
    5: {
        "title": "Hızlı Verimlilik Kaybı & Formasyon Tıkanması Müdahalesi",
        "action_level": "KRİTİK ÜRETİM KAYBI",
        "color": "#DC2626",
        "steps": [
            "Formasyon yakınında kirlenme/tıkanma (skin factor artışı) riskine karşı P-PDG basınç düşümünü analiz edin.",
            "Kuyu perforasyon bölgesindeki hidrolik geçirgenliği artırmak için asitleme (acidizing) veya solvent yıkama planlayın.",
            "Üretim hattı vanalarındaki olası yabancı madde birikimini ve filtreleri kontrol edin."
        ]
    },
    6: {
        "title": "PCK Vanasında Hızlı Daralma & Mekanik Kilitlenme Aksiyonu",
        "action_level": "KRİTİK MEKANİK MÜDAHALE",
        "color": "#B91C1C",
        "steps": [
            "Şok vana memba (P-MON-CKP) ve mansap (P-JUS-CKP) basınç farkını derhal ölçün.",
            "Vana aktüatör sinyalini ve hidrolik/elektrik konumlayıcıyı (positioner) manuel moda alıp yeniden kalibre edin.",
            "Vana gövdesindeki mekanik sıkışmaya karşı vana açıklığını %50 artırıp azaltarak pislik temizleme döngüsü uygulayın."
        ]
    },
    7: {
        "title": "PCK Vanasında Kireçlenme (Scaling) Kimyasal Müdahalesi",
        "action_level": "YÜKSEK KİMYASAL RİSK",
        "color": "#8B5CF6",
        "steps": [
            "Kireçlenme önleyici (Scale Inhibitor) kimyasal pompalama debisini 2 katına çıkarın.",
            "Vana memesindeki kalsiyum karbonat / baryum sülfat çökelmesini çözmek için solvent enjeksiyonunu başlatın.",
            "Sıcaklık ve basınç düşüşünün kireçlenme eşik değerinin altına inmemesi için akış ceketlerini aktif hale getirin."
        ]
    },
    8: {
        "title": "Üretim Hattında Gaz-Hidrat Kristalleşmesi Önleme Prosedürü",
        "action_level": "KRİTİK TIKANMA RİSKİ",
        "color": "#06B6D4",
        "steps": [
            "Üretim hattına MEG (Monoetilen Glikol) veya Methanol Termodinamik Hidrat İnhibitörü enjeksiyonunu derhal başlatın.",
            "Hat sıcaklığını (T-JUS-CKP) ortam gazının hidrasyon oluşum sıcaklığının (genelde >18°C) üzerine çıkarmak için ısıtma sistemini devreye alın.",
            "Tıkanmanın ilerlemesi durumunda kuyu akış hattını kademeli olarak depresürize (depressurize) ederek gaz birikimini boşaltın."
        ]
    },
    9: {
        "title": "Servis Hattında Hidrat Birikmesi Acil Müdahalesi",
        "action_level": "KRİTİK SERVİS HATTI MÜDAHALESİ",
        "color": "#3B82F6",
        "steps": [
            "Servis hattı giriş ve çıkış isolasyon vanalarını kontrol edin.",
            "Servis hattına yüksek basınçlı metanol basarak buzlaşan hidrat kristallerinin erimesini sağlayın.",
            "Hat üzerindeki basınç dalgalanması sıfırlanana kadar hat debisini kontrol altında tutun."
        ]
    }
}

def get_engineering_recommendations(class_id: int, top_features: list = None) -> dict:
    """
    Generates domain expert action steps based on diagnosed class and SHAP top sensor features.
    """
    rec = RECOMMENDATIONS_DATABASE.get(class_id, RECOMMENDATIONS_DATABASE[0]).copy()
    
    sensor_specific_notes = []
    if top_features:
        for item in top_features[:3]:
            feat_name = item.get('feature', '')
            if 'DELTA-P-CKP' in feat_name:
                sensor_specific_notes.append("⚠️ **DELTA-P-CKP**: Şok vana basınç farkı kritik dalgalanıyor; vana aşınmasını ve manifold girişini öncelikle denetleyin.")
            elif 'P-PDG' in feat_name:
                sensor_specific_notes.append("⚠️ **P-PDG**: Kuyu dibi basıncı doğrudan etkileniyor; rezervuar irtibatını ve kuyu dibi sensör kalibrasyonunu kontrol edin.")
            elif 'T-MON-CKP' in feat_name or 'T-JUS-CKP' in feat_name:
                sensor_specific_notes.append("⚠️ **T-CKP**: Sıcaklık anomalisi saptandı; hidrasyon veya kireçlenme kristalleşme sınırında olabilirsiniz.")
            elif 'P-ANULAR' in feat_name:
                sensor_specific_notes.append("⚠️ **P-ANULAR**: Anüler basınç değişimi tespiti; muhafaza borusu (casing) sızdırmazlık testini gerçekleştirin.")

    rec['sensor_notes'] = sensor_specific_notes
    return rec
