"""Model loading, chat template, read position (spec 0.4)."""
import hashlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_id, dtype=torch.bfloat16, device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                device_map=device_map)
    model.eval()
    return tok, model


def templated(tok, prompt):
    """User turn + generation prompt. The read position is its final token."""
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)


def token_ids(tok, prompt):
    return tok(templated(tok, prompt), add_special_tokens=False)["input_ids"]


def prompt_hasher(tok):
    """sha256 over token ids -> prompt_sha16 (spec 0.8)."""
    def hash_fn(prompt):
        ids = token_ids(tok, prompt)
        return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]
    return hash_fn


def token_info_fn(tok):
    """Final token id and length at the read position, recorded in the view.

    If the two poles systematically end on different tokens, a perfect AUROC is a
    token-identity readout and says nothing about the construct.
    """
    def info(prompt):
        ids = token_ids(tok, prompt)
        return {"n_tokens": len(ids), "last_token_id": int(ids[-1])}
    return info


def chat_template_sha(tok):
    tpl = getattr(tok, "chat_template", None) or ""
    return hashlib.sha256(tpl.encode()).hexdigest()[:16]


@torch.no_grad()
def last_token_hidden(tok, model, prompts, batch_size=8):
    """[n, L+1, d] fp16 at the final prompt token. Left padding, so index -1."""
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = [templated(tok, p) for p in prompts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        # Left padding puts every real last token at -1; assert rather than trust.
        assert enc["attention_mask"][:, -1].all(), "right padding would read a pad token"
        hs = model(**enc, output_hidden_states=True).hidden_states
        out.append(torch.stack([h[:, -1, :] for h in hs], dim=1).to(torch.float16).cpu())
    return torch.cat(out).numpy()
