import json
import os

def load_prompt_profile():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "bark_tagging_profile.json"), "r") as f:
        profile = json.load(f)
    with open(os.path.join(base, "bark_emotion_tag_mapping.json"), "r") as f:
        emotion = json.load(f)
    return profile, emotion