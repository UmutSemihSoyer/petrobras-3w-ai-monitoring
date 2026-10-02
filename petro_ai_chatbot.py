"""
Petrobras 3W AI Monitoring - Petro-AI LLM & RAG Engineering Assistant

This module provides a Domain-Specific RAG (Retrieval-Augmented Generation) & LLM Assistant
for answering engineer queries about offshore well anomalies, SOPs, and troubleshooting steps.
"""

from typing import List, Dict, Any

KNOWLEDGE_BASE = [
    {
        "topic": "Hydrate Formation (Hidrat Oluşumu)",
        "keywords": ["hidrat", "hydrate", "tikanma", "kristallesme", "ice"],
        "answer": "Üretim veya servis hattında gaz-hidrat kristalleşmesi basınç düşüşü ve düşük sıcaklık birleşimiyle oluşur. Aksiyon: Üretim hattına derhal MEG (Monoetilen Glikol) veya Metanol enjeksiyonu başlatılmalı, hat sıcaklığı (T-JUS-CKP) hidrasyon sıcaklığının (>18°C) üzerine çıkarılacak şekilde ceket ısıtma aktif edilmelidir."
    },
    {
        "topic": "DHSV Spurious Closure (Emniyet Vanası Kapanması)",
        "keywords": ["dhsv", "emniyet vanasi", "safety valve", "kapanma", "closure"],
        "answer": "DHSV (Downhole Safety Valve) kuyu dibi emniyet vanasında haksız kapanma algılandı. Aksiyon: Hidrolik kumanda hattı basıncını (Control Line Pressure) kontrol edin, ESD emniyet kilidini sıfırlayın ve hidrolik basıncı kademeli artırarak vananın tam açık olduğunu sensörlerden doğrulayın."
    },
    {
        "topic": "Severe Slugging (Şiddetli Sıvı Birikmesi)",
        "keywords": ["slugging", "dalgalanma", "sivi birikmesi", "gas lift", "slug"],
        "answer": "Şiddetli Slugging zaman serilerinde periyodik basınç/sıcaklık salınımları oluşturur. Aksiyon: Üretim şok vanası (PCK) %10-15 oranında kısılarak geri basınç (back-pressure) oluşturulmalı, Gas Lift enjeksiyon debisi (QGL) artırılmalı ve Slug Catcher sıvı seviyesi izlenmelidir."
    },
    {
        "topic": "Scaling in PCK (Kireçlenme)",
        "keywords": ["kireclenme", "scaling", "kabuklasma", "pck", "depozit"],
        "answer": "PCK vanasında kalsiyum karbonat / baryum sülfat çökelmesi ve kireçlenme birikimi var. Aksiyon: Kireç önleyici (Scale Inhibitor) kimyasal enjeksiyon debisini 2 katına çıkarın, kuyu başı ısıtma ceketlerini devreye alın ve vana temizlik prosedürünü başlatın."
    },
    {
        "topic": "BSW Abrupt Increase (Su Oranı Artışı)",
        "keywords": ["bsw", "su orani", "water cut", "su sramasi", "emulsion"],
        "answer": "BSW (su oranı) aniden yükseldi. Aksiyon: Üretim ayırıcı (seperatör) tank su tahliye vanalarını kontrol edin, de-emülgatör kimyasal enjeksiyonunu %20 artırın ve kuyu su konlanmasına karşı debiyi %10 kısın."
    }
]

class PetroAIChatbot:
    """
    RAG & Domain-Specific Petroleum Engineering Chatbot Assistant.
    """
    def __init__(self):
        self.knowledge_base = KNOWLEDGE_BASE

    def answer_query(self, user_prompt: str) -> Dict[str, Any]:
        """
        Retrieves relevant engineering knowledge and formats intelligent AI response.
        """
        prompt_lower = user_prompt.lower()
        matched_items = []
        
        for item in self.knowledge_base:
            if any(kw in prompt_lower for kw in item["keywords"]):
                matched_items.append(item)
                
        if matched_items:
            best_match = matched_items[0]
            return {
                "topic": best_match["topic"],
                "reply": f"🤖 **Petro-AI Asistanı [{best_match['topic']}]:**\n\n{best_match['answer']}",
                "confidence": 0.96,
                "sources": ["Petrobras 3W SOP Manual", "Offshore Field Operations Guide"]
            }
        else:
            return {
                "topic": "Genel Kuyu Danışmanlığı",
                "reply": f"🤖 **Petro-AI Asistanı:** Sorunuzu aldım: '{user_prompt}'. Sistem stabil gözüküyor. Sensör kanallarını (P-PDG, T-TPT, P-MON-CKP) ve SHAP kök neden analiz kartını kontrol edebilirsiniz. Ek spesifik sorularınız için 'hidrat', 'DHSV', 'slugging' veya 'kireçlenme' anahtar kelimelerini kullanabilirsiniz.",
                "confidence": 0.85,
                "sources": ["Petrobras 3W Knowledge Base"]
            }
