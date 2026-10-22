"""Model adapters for the VLM smoke test.

Each adapter exposes load() -> dict of loaded objects, and
predict(loaded, image, prompt) -> str. Models are loaded one at a time in
run_eval.py (and freed after use) to keep memory usage manageable.
"""
import torch


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype():
    return torch.float16 if torch.cuda.is_available() else torch.float32


# Plain greedy decoding (the default) tends to loop into repetitive text on
# these models for open-ended prompts — these kwargs discourage that without
# switching to sampling (which would make answers non-reproducible).
GENERATE_KWARGS = {"max_new_tokens": 100, "repetition_penalty": 1.3, "no_repeat_ngram_size": 3}


def _load_on_device(model_cls, model_id):
    """Load a HF causal-VLM, avoiding accelerate's device_map="auto".

    device_map="auto" is meant for juggling multiple GPUs (or GPU+CPU
    offload) and on a single CPU-only machine with limited RAM it can
    decide to "offload" some weights to disk/meta-device — which then
    segfaults instead of raising a catchable error the first time those
    weights are actually used. Loading straight onto one device skips that
    path: it either fits in RAM or fails with a normal MemoryError.
    """
    if torch.cuda.is_available():
        return model_cls.from_pretrained(model_id, torch_dtype=_dtype(), device_map="auto")
    model = model_cls.from_pretrained(model_id, torch_dtype=_dtype(), low_cpu_mem_usage=True)
    model.to("cpu")
    return model


# ---------- LLaVA-1.5 (7B) — Kaggle GPU only, too large for a CPU laptop ----------

def load_llava():
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = _load_on_device(LlavaForConditionalGeneration, model_id)
    return {"model": model, "processor": processor}


def predict_llava(loaded, image, prompt):
    model, processor = loaded["model"], loaded["processor"]
    chat = f"USER: <image>\n{prompt} ASSISTANT:"
    inputs = processor(text=chat, images=image, return_tensors="pt").to(model.device, model.dtype)
    output = model.generate(**inputs, **GENERATE_KWARGS)
    text = processor.decode(output[0], skip_special_tokens=True)
    return text.split("ASSISTANT:")[-1].strip()


# ---------- Qwen2-VL — 7B (Kaggle GPU only) and 2B (local-friendly) ----------

def _load_qwen2vl(model_id):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    model = _load_on_device(Qwen2VLForConditionalGeneration, model_id)
    return {"model": model, "processor": processor}


def load_qwen2vl_7b():
    return _load_qwen2vl("Qwen/Qwen2-VL-7B-Instruct")


def load_qwen2vl_2b():
    return _load_qwen2vl("Qwen/Qwen2-VL-2B-Instruct")


def predict_qwen2vl(loaded, image, prompt):
    model, processor = loaded["model"], loaded["processor"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device, model.dtype)
    output = model.generate(**inputs, **GENERATE_KWARGS)
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    return decoded.split("assistant\n")[-1].strip()


# ---------- BLIP-2 ----------

def load_blip2():
    from transformers import AutoProcessor, Blip2ForConditionalGeneration

    model_id = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_id)
    model = _load_on_device(Blip2ForConditionalGeneration, model_id)
    return {"model": model, "processor": processor}


def predict_blip2(loaded, image, prompt):
    # BLIP-2 (the non-instruct "opt" checkpoint) is a continuation model, not
    # a chat model: given an open-ended prompt it tends to just continue/echo
    # it rather than answer it. It needs the "Question: ... Answer:" framing
    # it was actually trained on to produce a real answer.
    model, processor = loaded["model"], loaded["processor"]
    qa_prompt = f"Question: {prompt} Answer:"
    inputs = processor(images=image, text=qa_prompt, return_tensors="pt").to(model.device, model.dtype)
    output = model.generate(**inputs, **GENERATE_KWARGS)
    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    return decoded[len(qa_prompt):].strip()


# ---------- Zoobot (astronomy baseline) ----------
# Uses only the pretrained encoder (via timm), not a GZ2-decision-tree
# classification head, so predict_zoobot() returns an embedding-norm
# placeholder rather than a real morphology prediction. Getting real
# GZ2-style predictions needs a specific finetuned head checkpoint that
# wasn't verified before writing this — flagged rather than guessed at.

def load_zoobot():
    import timm

    model = timm.create_model("hf_hub:mwalmsley/zoobot-encoder-convnext_nano", pretrained=True)
    model.eval()
    model.to(_device())
    return {"model": model}


def predict_zoobot(loaded, image, prompt):
    import torchvision.transforms as T

    model = loaded["model"]
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(image.convert("RGB")).unsqueeze(0).to(_device())
    with torch.no_grad():
        features = model(x)
    return f"PLACEHOLDER (encoder embedding only, no morphology head): embedding_norm={features.norm().item():.3f}"


# ---------- AstroCLIP (astronomy baseline) ----------
# Deliberately disabled by default. The only public weights found
# (giovannicozzolongo/astroclip on Hugging Face) require
# trust_remote_code=True, which runs a Python file from that repo on this
# machine. That's a decision for a human to make explicitly, not a default
# to flip quietly — set ALLOW_ASTROCLIP_REMOTE_CODE = True yourself once
# you're comfortable with that, or swap in the official
# PolymathicAI/AstroCLIP checkpoint + loading code instead.
ALLOW_ASTROCLIP_REMOTE_CODE = False


def load_astroclip():
    if not ALLOW_ASTROCLIP_REMOTE_CODE:
        raise RuntimeError(
            "AstroCLIP's only public weights (giovannicozzolongo/astroclip) need "
            "trust_remote_code=True to load, which runs that repo's code on this "
            "machine. Set models.ALLOW_ASTROCLIP_REMOTE_CODE = True once that's "
            "okay with you, or switch to the official PolymathicAI/AstroCLIP "
            "checkpoint + loading code instead."
        )
    from transformers import AutoModel

    model = AutoModel.from_pretrained("giovannicozzolongo/astroclip", trust_remote_code=True)
    model.eval()
    model.to(_device())
    return {"model": model}


def predict_astroclip(loaded, image, prompt):
    # Exact preprocessing/forward signature not verified against the real
    # checkpoint yet — likely needs adjusting on first real run. Returns an
    # embedding-norm placeholder, same spirit as Zoobot: a baseline, not a
    # chat-style answer.
    import torchvision.transforms as T

    model = loaded["model"]
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(image.convert("RGB")).unsqueeze(0).to(_device())
    with torch.no_grad():
        features = model(x)
    return f"PLACEHOLDER (embedding only): embedding_norm={features.norm().item():.3f}"


MODEL_REGISTRY = {
    "llava": (load_llava, predict_llava),              # Kaggle GPU only
    "qwen2vl_7b": (load_qwen2vl_7b, predict_qwen2vl),   # Kaggle GPU only
    "qwen2vl_2b": (load_qwen2vl_2b, predict_qwen2vl),   # local-friendly
    "blip2": (load_blip2, predict_blip2),               # local-friendly (slow on CPU)
    "zoobot": (load_zoobot, predict_zoobot),            # local-friendly, placeholder output
    "astroclip": (load_astroclip, predict_astroclip),   # disabled until remote-code decision is made
}
