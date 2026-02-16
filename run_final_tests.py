"""Final hackathon test: all 5 files against legacy POST /api/voice-detection"""
import base64, json, time, requests

DIR = r"c:\Users\shiva\OneDrive\Desktop\Voice Project\voice-detection-api\drive-download-20260216T053632Z-1-001"
URL = "http://localhost:7860/api/voice-detection"
HEADERS = {"Content-Type": "application/json", "x-api-key": "sk_test_voice_detection_2026"}

FILES = [
    ("English_voice_AI_GENERATED.mp3", "English", "AI_GENERATED"),
    ("Hindi_Voice_HUMAN.mp3", "Hindi", "HUMAN"),
    ("Malayalam_AI_GENERATED.mp3", "Malayalam", "AI_GENERATED"),
    ("TAMIL_VOICE__HUMAN.mp3", "Tamil", "HUMAN"),
    ("Telugu_Voice_AI_GENERATED.mp3", "Telugu", "AI_GENERATED"),
]

print("=" * 90)
print(f"{'File':<42} {'Expected':<16} {'Got':<16} {'Conf':>6}  Result")
print("=" * 90)

passed = 0
for fname, lang, expected in FILES:
    with open(f"{DIR}\\{fname}", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {"audioBase64": b64, "language": lang, "audioFormat": "mp3"}
    t0 = time.time()
    try:
        r = requests.post(URL, json=payload, headers=HEADERS, timeout=30)
        elapsed = time.time() - t0
        d = r.json()
        cls = d.get("classification", "?")
        conf = d.get("confidenceScore", "?")
        ok = cls == expected
        if ok:
            passed += 1
        tag = "PASS" if ok else "FAIL"
        print(f"{fname:<42} {expected:<16} {cls:<16} {conf:>6}  {tag}  ({elapsed:.1f}s)")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{fname:<42} {expected:<16} {'ERROR':<16} {'--':>6}  FAIL  ({elapsed:.1f}s) {e}")
    # small pause between requests to avoid CPU thermal throttle
    time.sleep(2)

print("=" * 90)
print(f"Result: {passed}/{len(FILES)} passed")
