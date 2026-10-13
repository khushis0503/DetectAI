"""
api_scenario_test.py - Sends five scenario texts and four invalid inputs to the running DetectAI
server (http://127.0.0.1:8000) and prints the verdicts. Start run.bat first.
"""
import json, urllib.request, urllib.error
import pandas as pd

def call(path, body=None):
    req = urllib.request.Request("http://127.0.0.1:8000" + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

df = pd.read_csv("data/detectai_dataset.csv")
test = df[(df.split == "test") & (df.n_words >= 70)]
samples = ("To further clarify: pi", "China said on Thursday its jailing", "Steam is a digital")
test = test[~test.text.str.startswith(samples)]
formal = test[(test.domain == "reuter") & (test.label == 0)].sample(1, random_state=1).iloc[0]
gpt = test[test.label == 1].sample(1, random_state=1).iloc[0]
half_h = test[(test.label == 0) & (test.domain == "essay")].sample(1, random_state=2).iloc[0]
half_a = test[(test.label == 1) & (test.domain == "essay")].sample(1, random_state=2).iloc[0]
short_src = test.sample(1, random_state=3).iloc[0]

cases = [
    ("1. 20-word text", " ".join(short_src.text.split()[:20]), f"{short_src.source}, {short_src.domain}"),
    ("2. Casual (yours)", "honestly i didnt get time to finish the assignment yesterday because the wifi kept going off and my laptop was slow", "human"),
    ("3. Formal human (Reuters)", formal.text, "human, reuter"),
    ("4. ChatGPT paragraph", gpt.text, f"ChatGPT, {gpt.domain} ({gpt.gen_style})"),
    ("5. Half human + half AI", half_h.text + "\n\n" + half_a.text, "para 1 human essay, para 2 ChatGPT essay"),
]
print("health:", call("/api/health")[1])
for name, text, truth in cases:
    status, r = call("/api/predict", {"text": text})
    print(f"\n=== {name}  [truth: {truth}]  HTTP {status}")
    print("TEXT:", text[:230].replace("\n", " / "), "..." if len(text) > 230 else "")
    print(f"verdict={r['verdict']}  P(AI)={r['ai_probability']:.3f}  confidence={r['confidence']:.3f}  "
          f"words={r['n_words']}  short_warning={r['short_text_warning']}  time={r['seconds']}s")
    top = sorted([s for s in r["highlighted"] if s["s"] is not None], key=lambda s: s["s"])
    print("  strongest -> human:", [s["t"] for s in top[:5]], "| -> AI:", [s["t"] for s in top[::-1][:5]])
    print("  sentences:", [round(s["ai_probability"], 2) for s in r["sentences"]])
    for b in r["breakdown"]:
        print(f"  {b['label']}: P(AI)={b['ai_probability']:.3f} -> {b['verdict']}  ({b['n_words']} words)")
    st = r["style"]
    print(f"  style: sent_len_std={st['sent_len_std']:.1f} TTR={st['type_token_ratio']:.2f} nouns={st['noun_ratio']:.2f} pron={st['pron_ratio']:.2f}")

print("\n=== validation errors")
for body in [{"text": ""}, {"text": "too short here"}, {"text": "word " * 3001}, {}]:
    print(call("/api/predict", body))
info = call("/api/model-info")[1]
print("\nmodel-info:", {k: info[k] for k in ["model", "model_name", "ensemble_used", "thresholds", "thresholds_calibrated"]})
