"""InteracVid causality probe v2 — more data + an LLM-as-judge.

Same premise as v1 (a response is caused by the query), strengthened: ~20 examples
across more scenarios, a stronger model (Qwen2.5-3B-Instruct), and the model itself
scores 1-5 how well each reply answers the query given the scene (LLM-as-judge),
compared for REAL vs SHUFFLED query. Emits a RESULTS_JSON line for eval_harness.
"""
import re, json, random

MODEL = "Qwen/Qwen2.5-3B-Instruct"

DATA = [
    ("A streamer is dicing onions for a curry.", "how do you stop crying when cutting onions?", "chill the onion first and use a sharp knife"),
    ("A streamer is browning chicken in a pan.", "can I use tofu instead of chicken?", "yes, press the tofu and sear it longer so it holds"),
    ("A streamer is unboxing a mechanical keyboard.", "are those switches loud?", "these are tactile browns, quieter than clicky blues"),
    ("A streamer is unboxing a camera.", "does it come with a memory card?", "no, just the body and strap, buy the card separately"),
    ("A streamer is doing a yoga flow.", "my knee hurts in this pose, what do I do?", "drop the back knee to the mat with a towel under it"),
    ("A streamer is on a platformer boss fight.", "how did you dodge that laser?", "dash on the second flash, not the first, that's the tell"),
    ("A streamer is reviewing a coffee grinder.", "is it too loud for early mornings?", "it's a burr grinder, a low hum, fine for mornings"),
    ("A streamer is sketching a portrait live.", "what pencil are you using for the shadows?", "a soft 6B for dark shadows, 2B for mid tones"),
    ("A streamer is potting a monstera plant.", "how often should I water it?", "once the top inch of soil is dry, about weekly"),
    ("A streamer is tuning a guitar.", "why does the low string sound off?", "the low E is flat, tighten it until it matches"),
    ("A streamer is doing a skincare routine.", "should serum go before or after moisturizer?", "serum first, then lock it in with moisturizer"),
    ("A streamer is soldering a circuit board.", "why won't the joint stick?", "the pad is dirty, clean it and add fresh flux"),
    ("A streamer is baking sourdough.", "why is my loaf so flat?", "the dough was over-proofed, shorten the final rise"),
    ("A streamer is setting up a tent on a livestream.", "will it hold up in the rain?", "yes, seam-seal it and stake the rainfly taut"),
    ("A streamer is doing calligraphy.", "what nib is that for thin lines?", "an EF nib, and lighten your pressure on upstrokes"),
    ("A streamer is fixing a bike.", "the brakes feel spongy, what's wrong?", "there's air in the line, you need to bleed the brakes"),
    ("A streamer is mixing a cocktail.", "can I make this without alcohol?", "sure, swap the gin for tonic and add more lime"),
    ("A streamer is knitting a scarf.", "how do I stop the edges curling?", "add a garter stitch border on each side"),
    ("A streamer is reviewing headphones.", "is the bass too heavy for podcasts?", "the bass is strong but voices still stay clear"),
    ("A streamer is doing makeup.", "how do I make lipstick last longer?", "blot, add powder, then a second thin layer"),
]


def token_f1(pred, gold):
    p = re.findall(r"[a-z0-9]+", pred.lower()); g = re.findall(r"[a-z0-9]+", gold.lower())
    if not p or not g: return 0.0
    ps, gs = set(p), set(g); inter = len(ps & gs)
    if inter == 0: return 0.0
    prec, rec = inter/len(ps), inter/len(gs)
    return 2*prec*rec/(prec+rec)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading", MODEL, flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")

    def gen(messages, max_new=40):
        enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def respond(scene, query):
        return gen([{"role": "system", "content": "You are the streamer. Reply in one short spoken sentence."},
                    {"role": "user", "content": f"Scene: {scene}\nViewer: {query}\nYour reply:"}])

    def judge(scene, query, reply):
        r = gen([{"role": "system", "content": "You are a strict evaluator. Answer with only a single digit 1-5."},
                 {"role": "user", "content": f"Scene: {scene}\nViewer question: {query}\nStreamer reply: {reply}\n"
                                             "How well does the reply answer the viewer's question in this scene? 1=irrelevant, 5=perfect. Answer 1-5:"}], max_new=3)
        m = re.search(r"[1-5]", r)
        return (int(m.group())/5.0) if m else 0.0

    rng = random.Random(0)
    queries = [q for _, q, _ in DATA]
    real_judge, ctrl_judge, real_f1, ctrl_f1 = [], [], [], []
    for i, (scene, q, gold) in enumerate(DATA):
        wrong = rng.choice([x for j, x in enumerate(queries) if j != i])
        rr, cc = respond(scene, q), respond(scene, wrong)
        real_judge.append(judge(scene, q, rr)); ctrl_judge.append(judge(scene, q, cc))
        real_f1.append(token_f1(rr, gold)); ctrl_f1.append(token_f1(cc, gold))
        print(f"[{i:2}] judge real={real_judge[-1]:.1f} ctrl={ctrl_judge[-1]:.1f} | REAL: {rr[:60]!r}", flush=True)

    res = {"real_judge": real_judge, "control_judge": ctrl_judge, "real_f1": real_f1, "control_f1": ctrl_f1,
           "n": len(DATA), "model": MODEL}
    print("\nRESULTS_JSON:", json.dumps(res))
    jr = sum(real_judge)/len(real_judge); jc = sum(ctrl_judge)/len(ctrl_judge)
    print(f"judge  real={jr:.3f} ctrl={jc:.3f} lift={jr-jc:+.3f}")
    print(f"f1     real={sum(real_f1)/len(real_f1):.3f} ctrl={sum(ctrl_f1)/len(ctrl_f1):.3f}")
    print("PROBE_DONE")


main()
