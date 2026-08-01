"""Render-time prompt construction. Single home of the preamble (spec 1.2a).

The same string is declared in data/story_mode_v2/build_dataset.py and
data/story_mode/build_matched_dataset.py. It must not drift: the v2-vs-v1
comparison is only a comparison if all render sites agree byte for byte.
"""

PREAMBLE = "Continue the text below."


def continuation(text):
    return f"{PREAMBLE}\n\n{text}"


def with_task(framing, task):
    """Spec 0.2(a): a framing-only prompt followed by a base task.

    The task is byte-identical across the pair, so the contrast stays
    framing-only while the read position sits after a request.
    """
    return f"{framing.rstrip()}\n\n{task}"
