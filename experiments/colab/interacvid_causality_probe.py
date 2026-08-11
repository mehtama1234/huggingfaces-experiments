"""InteracVid causality probe — does a response actually depend on the query?

Tests InteracVid's central premise (a reaction is *caused by* a stimulus, not just
decoration) with a real small instruct model on a T4. For each (context, query,
gold_response) we ask the model to produce the response two ways:
  REAL    : given context + the real query
  CONTROL : given context + a shuffled query from another example
and score each generation's token-F1 against the gold response. If REAL > CONTROL,
the query genuinely drives the response — the signal InteracVid says caption data lacks.
"""
import re, random, sys

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Hand-authored tiny set across 3 InteracVid-style scenarios (context, query, gold response).
DATA = [
    ("A streamer is dicing onions for a curry.", "how do you stop crying when cutting onions?",
     "chill the onion first and use a sharp knife so you crush fewer cells"),
    ("A streamer is browning chicken in a pan.", "can I use tofu instead of chicken?",
     "yes, press the tofu and sear it a bit longer so it holds together"),
    ("A streamer is unboxing a new mechanical keyboard.", "are those switches loud?",
     "these are tactile browns, so they're quieter than clicky blues"),
    ("A streamer is unboxing a camera.", "does it come with a memory card?",
     "no, the box only has the body and a strap, you buy the card separately"),
    ("A streamer is doing a yoga flow.", "my knee hurts in this pose, what do I do?",
     "drop the back knee to the mat and put a folded towel under it"),
    ("A streamer is playing a platformer boss fight.", "how did you dodge that laser?",
     "you dash on the second flash, not the first, that's the tell"),
    ("A streamer is reviewing a coffee grinder.", "is it too loud for early mornings?",
     "it's a burr grinder so it's a low hum, fine for early mornings"),
    ("A streamer is sketching a portrait live.", "what pencil are you using for the shadows?",
     "a soft 6B for the dark shadows and a 2B for the mid tones"),
]


def token_f1(pred: str, gold: str) -> float:
    p = re.findall(r"[a-z0-9]+", pred.lower()); g = re.findall(r"[a-z0-9]+", gold.lower())
    if not p or not g: return 0.0
    ps, gs = set(p), set(g); inter = len(ps & gs)
    if inter == 0: return 0.0
    prec, rec = inter/len(ps), inter/len(gs)
    return 2*prec*rec/(prec+rec)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading", MODEL, "…", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="cuda")

    def respond(context, query):
        msg = [{"role": "system", "content": "You are the streamer. Reply in one short spoken sentence."},
               {"role": "user", "content": f"Scene: {context}\nViewer: {query}\nYour reply:"}]
        enc = tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        in_len = enc["input_ids"].shape[1]
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, in_len:], skip_special_tokens=True).strip()

    rng = random.Random(0)
    queries = [q for _, q, _ in DATA]
    real_f1, ctrl_f1 = [], []
    for i, (ctx, q, gold) in enumerate(DATA):
        wrong = rng.choice([x for j, x in enumerate(queries) if j != i])  # a shuffled query
        r = respond(ctx, q); c = respond(ctx, wrong)
        rf, cf = token_f1(r, gold), token_f1(c, gold)
        real_f1.append(rf); ctrl_f1.append(cf)
        print(f"[{i}] real_f1={rf:.2f} ctrl_f1={cf:.2f} | REAL: {r[:70]!r}", flush=True)

    mr, mc = sum(real_f1)/len(real_f1), sum(ctrl_f1)/len(ctrl_f1)
    print(f"\nREAL query mean token-F1 : {mr:.3f}")
    print(f"SHUFFLED query mean F1   : {mc:.3f}")
    print(f"causality lift (real-ctrl): {mr-mc:+.3f}")
    print("verdict:", "query DRIVES the response (causal signal present)" if mr > mc + 0.02
          else "little query sensitivity")
    print("PROBE_DONE")


main()
