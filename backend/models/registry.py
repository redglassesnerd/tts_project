import logging
from typing import Dict, List
from models.impl import (
    VITSTTS,
    BarkTTS,
    XTTSv2,
    KokoroTTS,
    Qwen3TTS,
    ChatterboxTTS,
    VibeVoiceTTS,
    CosyVoiceTTS,
    ChatTTSTTS,
    FishAudioTTS,
)

logger = logging.getLogger("backend.models.registry")

class ModelRegistry:
    def __init__(self):
        self._registry = {}
        self._short_names = {}
        
        # Instantiate model handlers
        vits = VITSTTS()
        bark = BarkTTS()
        xtts = XTTSv2()
        kokoro = KokoroTTS()
        qwen = Qwen3TTS()
        chatterbox = ChatterboxTTS()
        vibe = VibeVoiceTTS()
        cosy = CosyVoiceTTS()
        chattts = ChatTTSTTS()
        fish = FishAudioTTS()

        # Register by full identifier name
        self.register(vits.get_metadata()["name"], vits)
        self.register(bark.get_metadata()["name"], bark)
        self.register(xtts.get_metadata()["name"], xtts)
        self.register(kokoro.get_metadata()["name"], kokoro)
        self.register(qwen.get_metadata()["name"], qwen)
        self.register(chatterbox.get_metadata()["name"], chatterbox)
        self.register(vibe.get_metadata()["name"], vibe)
        self.register(cosy.get_metadata()["name"], cosy)
        self.register(chattts.get_metadata()["name"], chattts)
        self.register(fish.get_metadata()["name"], fish)

        # Register by short alias keys for easier API routing/backward compatibility
        self.register_alias("vits", vits.get_metadata()["name"])
        self.register_alias("bark", bark.get_metadata()["name"])
        self.register_alias("xtts", xtts.get_metadata()["name"])
        self.register_alias("kokoro", kokoro.get_metadata()["name"])
        self.register_alias("qwen3-tts", qwen.get_metadata()["name"])
        self.register_alias("chatterbox-turbo", chatterbox.get_metadata()["name"])
        self.register_alias("vibevoice", vibe.get_metadata()["name"])
        self.register_alias("cosyvoice2-styletts2", cosy.get_metadata()["name"])
        self.register_alias("chattts", chattts.get_metadata()["name"])
        self.register_alias("fish-audio", fish.get_metadata()["name"])

    def register(self, name: str, model_instance):
        self._registry[name.lower()] = model_instance

    def register_alias(self, alias: str, target_name: str):
        self._short_names[alias.lower()] = target_name.lower()

    def get(self, name: str):
        if not name:
            return None
        key = name.lower()
        # Resolve alias if present
        if key in self._short_names:
            key = self._short_names[key]
        return self._registry.get(key)

    def preload_all(self):
        logger.info("Preloading registered TTS models...")
        for name, model in self._registry.items():
            logger.info(f"Preloading model '{name}'...")
            try:
                success = model.preload()
                if not success:
                    model.is_simulator = True
                    logger.warning(f"Model '{name}' preload failed (returned False). Marked as simulator.")
            except Exception as e:
                model.is_simulator = True
                logger.error(f"Model '{name}' preload crashed: {e}. Marked as simulator.", exc_info=True)

    def get_all_metadata(self) -> List[dict]:
        # Gather all unique metadata configurations
        metadata_list = []
        seen = set()
        for name, model in self._registry.items():
            meta = model.get_metadata()
            if meta["name"] not in seen:
                seen.add(meta["name"])
                
                # Check if currently running in simulator mode
                meta["is_simulator"] = model.is_simulator
                metadata_list.append(meta)
        return metadata_list

# Singleton instance of the registry
MODEL_REGISTRY = ModelRegistry()
