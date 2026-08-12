"""Few-step latency/quality sweep — the "real-time = tiny step count" trick.

Vidu S1, InteractiveAvatar, AlayaWorld and DreamX all get real-time by distilling
generation down to a handful of steps. This measures that trade directly on a real
diffusion model (SD 1.5 on a T4): generate the SAME prompt at 1/2/4/8/16/32 steps,
timing each (latency -> effective FPS) and scoring image/prompt agreement with CLIP.
Emits RESULTS_JSON for eval_harness/Scorecard.
"""
import time, json, subprocess, sys

def _ensure(pkg, imp=None):
    try: __import__(imp or pkg)
    except ImportError: subprocess.run([sys.executable,"-m","pip","install","-q",pkg])

_ensure("diffusers"); _ensure("transformers"); _ensure("accelerate")

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import CLIPModel, CLIPProcessor

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPTS = ["a red fox sitting in fresh snow, photo",
           "a bowl of ramen on a wooden table, steam rising"]
STEPS = [1, 2, 4, 8, 16, 32]

def main():
    print("loading", MODEL, flush=True)
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    cproc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def clip_score(img, prompt):
        ins = cproc(text=[prompt], images=img, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = clip(**ins)
        ie = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        te = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((ie @ te.T)[0, 0])

    rows = {}
    for n in STEPS:
        lat, qual = [], []
        for p in PROMPTS:
            g = torch.Generator("cuda").manual_seed(0)
            torch.cuda.synchronize(); t = time.time()
            img = pipe(p, num_inference_steps=n, guidance_scale=7.0, generator=g).images[0]
            torch.cuda.synchronize(); dt = time.time() - t
            lat.append(dt); qual.append(clip_score(img, p))
        ml, mq = sum(lat)/len(lat), sum(qual)/len(qual)
        rows[n] = {"latency_s": ml, "fps": 1.0/ml, "clip": mq}
        print(f"steps={n:2}  latency={ml:5.2f}s  fps={1.0/ml:5.2f}  clip={mq:.3f}", flush=True)

    print("\nRESULTS_JSON:", json.dumps(rows))
    print("PROBE_DONE")

main()
