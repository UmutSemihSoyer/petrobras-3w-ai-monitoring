"""
Petrobras 3W AI Monitoring - Voice AI Assistant & Speech Command Processor

This module parses natural language voice queries (Whisper / Speech API stili)
from offshore field engineers into platform commands and AI queries.
"""

from typing import Dict, Any

class VoiceAssistantService:
    """
    Voice AI Command Processor for Petrobras Platform Engineers.
    """
    def process_voice_command(self, spoken_transcript: str) -> Dict[str, Any]:
        """
        Parses spoken text transcript into actionable API command intent.
        """
        clean_text = spoken_transcript.strip().lower()
        intent = "UNKNOWN"
        target = "GENERAL"
        response_speech = "Komut anlaşılamadı. Lütfen tekrar edin."

        if "kuyu" in clean_text or "well" in clean_text or "durum" in clean_text:
            intent = "GET_WELL_STATUS"
            target = "FLEET_MONITOR"
            response_speech = "10 adet platform kuyusunun canlı sağlık skorları ekrana getiriliyor."
        elif "pdf" in clean_text or "rapor" in clean_text or "indir" in clean_text:
            intent = "EXPORT_PDF_REPORT"
            target = "DIAGNOSTIC_REPORT"
            response_speech = "Son teşhis için PDF raporu hazırlanıyor ve indiriliyor."
        elif "acil" in clean_text or "esd" in clean_text or "kapat" in clean_text:
            intent = "TRIGGER_ESD_LOCK"
            target = "SAFETY_SYSTEM"
            response_speech = "DİKKAT: Acil durum emniyet kilit sistemi sorgulanıyor."
        elif "yardım" in clean_text or "neden" in clean_text or "hidrat" in clean_text:
            intent = "ASK_PETRO_AI"
            target = "CHATBOT"
            response_speech = "Petro-AI teknik kılavuzu sorgulanıyor."

        return {
            "transcript": spoken_transcript,
            "detected_intent": intent,
            "target_system": target,
            "response_text": response_speech,
            "success": intent != "UNKNOWN"
        }
