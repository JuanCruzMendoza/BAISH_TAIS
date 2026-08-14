"""Model loading, chat template, read position (spec 0.4)."""
import hashlib
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def tokenizer(model_id):
    """The tokenizer `load` would build, without the weights.

    Everything derived from the tokenizer alone -- `templated`, `prompt_hasher`,
    `token_info_fn`, `chat_template_sha`, and so a view's `view_key` -- is identical
    either way, which is what lets a view be built on CPU. Configured here rather than in
    `load` so the two can never drift: padding side and pad token both reach the hashes.
    """
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load(model_id, dtype=torch.bfloat16, device_map="auto"):
    """`$ATTN_IMPL` overrides the attention kernel; unset leaves transformers' own choice.

    Gemma-2 soft-caps the attention logits and the sdpa/flash kernels drop that, so it
    needs `eager` -- a different set of activations and a different generation, not a
    speed knob. An environment variable rather than an argument: every script here loads
    through this one call and none of them should have to know.
    """
    tok = tokenizer(model_id)
    impl = os.environ.get("ATTN_IMPL")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device_map,
        **({"attn_implementation": impl} if impl else {}))
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
