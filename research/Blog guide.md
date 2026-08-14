
**

# Blog Post Writing Guide

How to write the blog post for your empirical research project: what goes in each section, core principles, and a pre-publish checklist. For most readers, the post is all they will see of your work. Its quality decides whether your research is read, understood, and acted on. That makes the most important part of the project, not an afterthought.

  

Use it as a base. Deviate where your project calls for it.

  

Important: the blog post MUST be written in english.

## Principles

- Define your target audience. Decide who you are writing for before you start: AI safety researchers, ML practitioners without a safety background, or a wider technical audience. Calibrate jargon, definitions, framing, and motivation to that group.
    
- One main takeaway. Decide on the single thing you most want every reader to remember, and make sure the post points to it from start to finish. Even a reader who only skims should walk away with that one takeaway.
    
- Focus your effort on Figure 1 and the TL;DR. Most readers will only read the TL;DR and skim Figure 1, so most of your writing and design effort should go there.
    
- Write for a reader with zero context. You have months of project context; the reader has none. Every sentence should be easy to read without prior exposure to your project.
    
- Be as brief as possible. The reader's time and attention is the scarce resource you have to manage. If you can cut something without losing meaning, cut it.
    
- Plain language over jargon. Use jargon only where it is needed to precisely state your point, not to sound smart. Default to plain language. Err on the side of defining specialized terms: you generally do not need to define widely-known concepts like "transformer" or "LLM", but do define things like "sparse autoencoder" or "steering vector".
    
- Every sentence must be true. Fact-check every factual statement in the post, including prior work, numbers, definitions and your own results. No exaggerations. If you are not sure, verify it, qualify it, or drop it. A single false claim is enough to make a reader distrust the rest. Apply this to every sentence in every section, not only the key ones.
    
- Anticipate the skeptical reader. Imagine the reader actively looking for holes in your argument. Every claim should be one they can follow and check. Anticipate likely objections and address them in the text.
    
- Inform, do not persuade. Match your language to your evidence. Hedged language is a feature when warranted. Overclaiming is the fastest way to lose the readers you most want.
    
- Limitations up front. In AI safety, epistemic honesty matters a lot. A post with honest limitations is more useful than one that hides them.
    
- Figures do work prose cannot. A well-chosen figure can replace three paragraphs. Put effort into them.
    
- One layer at a time. Writing pulls on several cognitive tasks at once: building the argument, structuring the post, drafting prose, polishing wording, fixing grammar. Working memory can only hold a few items at a time, so trying to do all of these simultaneously means each is done worse and the whole process slows down. Run them as separate passes instead, with one layer per pass.
    
- Review in two passes. Review in two separate passes. The first pass is macro: argument, structure, flow between sections. The second is micro: word choice, grammar, citations. Doing both at once wastes time and mental energy.
    

  

## What is a claim?

According to [Neel Nanda](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers) a claim is a specific, evidence-backed statement of new knowledge that you want the reader to believe and remember. It is the unit of contribution: a post is built around one to three claims, and everything else exists to motivate and justify them.

  

Main properties of a good claim

- A contribution to the body of knowledge. A claim has to expand what is known. Replications and falsifications count as claims if they expand knowledge.
    
- Specific and concrete. Vague statements are not claims. Examples of claims: "A substantial part of the model's behavior in scenario A is explained by simple explanation B", "Technique C can fail in scenario D if conditions E and F hold".
    
- Backed by evidence. Each claim should be paired with the evidence the reader would need to believe it. "The essence of an ideal paper is the narrative: a short, rigorous and evidence-based technical story you tell, with a takeaway the readers care about."
    
- Coherent with the others. When you have several claims, they should fit a single theme, not a grab bag of disconnected findings.
    

  

Types of claim

  

Pick the type that matches the strength of your evidence. Stronger claims demand stronger evidence, and mismatching the two erodes trust.

  

- Existence-proof claim. "We found at least one example where X happens." The weakest form: you only need to show that X is possible, not that it is common.
    
- Systematic claim. "X generally happens across a wide range of contexts." Requires breadth: many examples, models, or settings.
    
- Hedged claim. "There is compelling / suggestive / tentative evidence that X is true." Use when the evidence points one way but does not pin it down. The hedge word should match the strength of the evidence.
    
- Narrow claim. "X is the best method in specific situations V and W, if your goal is to optimize Y." Restricts the scope explicitly, in exchange for a stronger statement within that scope.
    
- Guarantee. "X is always true." The strongest form. Almost never appropriate in deep learning, since universal statements require far more than experimental support.
    

## Writing order

Before you start writing:

  

1. Verify your critical experiments are correct.
    
2. Red-team yourself: assume you made a mistake somewhere. Where is it?
    

  

Then write the sections in a different order than the reader will read them. Each section is easier to write when the ones it depends on already exist.

  

Suggested order:
1. Define your core claims. Compress the project into 1–3 specific claims you want the reader to take away. For each claim, write one line of evidence and one line of why the reader should care. Everything else in the post supports these claims.
    
2. Outline. Map the post section by section, in bullets. Do not write prose yet.
    
3. Figures. Draft the key figures, especially Figure 1, before writing the prose around them. Figures often reveal what the prose needs to argue.
    
4. Results. Lead with the main result, then supporting ones.
    
5. Methods. Describe what you did at a level a competent reader could replicate.
    
6. Discussion. Interpret the results, list limitations, revisit the motivation.
    
7. Related Work. With your contribution clear, position it against 2–3 prior lines of work.
    
8. Introduction. Last among the long sections. You need to know where you ended up before you can write an effective roadmap.
    
9. TL;DR. Write this section last, once the rest of the post is drafted.
    

  

The title can be drafted at any point in the process.

## Structure

### Title

- The title should describe what your findings are, not just the topic.
    

### TL;DR (required)

A reader who sees only the TL;DR should know what the post argues and whether to keep reading.

  

Use a short, scannable bullet list. Each bullet is a self-contained statement. The reader should be able to understand it without clicking into the section. A useful TL;DR tends to include the following:

  

- The setup or existing state of the field, in one or two lines. What technique, problem, or assumption your work is reacting to.
    
- Motivation. Why this matters and what question the current situation leaves open.
    
- Your contributions. Your 1–3 novel, specific claims, ideally paired with the core evidence for each.
    
- Impact. How a reader should update their beliefs about the threat model or the broader safety picture.
    
- A link to the code, repo, or artifact.
    

  

Order the bullets so a reader can follow the story top to bottom.

### Figure 1 (required)

A "hero figure" near the top of the post that summarises your method or main result. Together with the title and the TL;DR, it should give a reader enough to grasp everything important about your work, even if they read nothing else. Put real effort into it.

  

Caption rule for every figure (including this one): state what the figure shows and the intended takeaway, not just a label.

### Introduction (required)

The introduction is broadly similar to the TLDR but more extended and in-depth. Treat it as a self-contained summary of the post: do not worry about "spoiling" later sections, since with a complex idea you want to repeat it several times until it sticks. The introduction is also where you have room to define the key terms and concepts the reader needs to follow your claims.

  

Cover these beats:

  

1. Context and motivation. The topic, the motivating question, and why it matters. For empirical AI safety, name the threat model and the specific failure mode.
    
2. Technical background. What is already known about the problem, which established techniques the work rests on and why prior attempts are inadequate.
    
3. Gap. What was not known, not tested, or contested before your work.
    
4. Research question. State it as a question.
    
5. Contribution. Your main claim, with nuance, context, and qualifications.
    
6. Your case. Briefly preview the strongest evidence behind the contribution. Do not save it for the Results section.
    
7. Impact. What the reader should take away, and how it applies to the broader safety picture.
    

  

If you have more than one main claim, repeat the Contribution and Your case beats once per claim.

  

By the end, a skeptical reader should know your claims, why they matter for safety and whether to keep reading.

### Methods (optional, recommended)

- Provide enough detail that a competent reader could replicate your work.
    
- Cover the models you used, the datasets, evaluation metrics, and any non-obvious setup choices. For hyperparameters, keep only the key ones in the main text and move the rest, along with any long prompts, to the appendix.
    
- Describe what you did, not the how of basic implementation. Skip details like "we wrote a Python script to process the CSV" or "we used a pandas merge". Those belong in the repo, not the post.
    

  

If you skip this section, include the essentials (models, datasets, evaluation metrics) into the Results section, so each result is presented alongside the method that produced it.

  

### Results (required)

- Lead with the main result that answers your research question, then put the supporting results after that.
    
- Prefer figures and tables over prose for quantitative findings. They make the results easier to visualize and remember than numbers buried in sentences.
    
- Report negative and null results and do not cherry-pick the ones that look best.
    
- If a negative result bears on your main claim, keep it in the main text. If it is supplementary (failed baselines, extended ablations), mention it briefly here and put the detail in the appendix.
    
- Use the prose around your figures and tables to interpret the results, not to repeat the numbers that the reader can already see.
    

### Discussion (required)

- Answer your research question. Be explicit about whether you answered it fully, partially, or not at all.
    
- Revisit the motivation. How should a reader update their beliefs about the threat model?
    
- Limitations. Include every relevant limitation. Do not omit the uncomfortable ones. What does your evidence not show? Where is your method fragile? What would break it?
    
- Calibrate your claims. Distinguish what you showed, what you believe but did not show, and what you speculate.
    

### Related Work (optional)

- Pick 2–3 lines of work most relevant to yours.
    
- For each: one line on what they did, one line on how your work differs or builds on them.
    

  

If you skip this section, add it into the Introduction.

### Future Work (optional, recommended)

- Suggest at least 2–3 concrete follow-ups. These should be specific experiments or extensions you consider relevant, not vague wish-lists.
    
- If you did not fully answer your question, describe what a next iteration would change.
    

### Acknowledgements (optional)

- Name who helped and how. Get consent before naming anyone publicly.
    

### Appendix (optional)

Put content in the appendix if any of the following hold:

  

- It supports a claim in the main post but is too long for the body (full hyperparameter tables, ablation grids, prompts).
    
- It helps a reader verify or reproduce your work (extended methodology, implementation detail).
    
- It reports additional experiments that are interesting or useful for follow-up but the main narrative does not depend on (exploratory findings, robustness checks, side ablations, failure cases).
    

  

In general, anything your main claims depend on belongs in the main text, not the appendix.

## Citations

Cite inline with hyperlinked author-year, APA-style. The link goes on the citation.

  

Examples:

  

- As in [Zou et al. 2023](https://arxiv.org/abs/2310.01405), we define a refusal direction as…
    
- Prior work ([Smith & Lee 2024](https://docs.google.com/document/d/1hsx66kHE9x8kEfP5PAUFSSnPi0K4jZgNgna5t5pX1rw/edit#); [Chen et al. 2025](https://docs.google.com/document/d/1hsx66kHE9x8kEfP5PAUFSSnPi0K4jZgNgna5t5pX1rw/edit#)) has shown…
    

## Pre-publish checklist

- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    Every sentence, in every section, has been verified for factual accuracy, with no exaggerations.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    Every sentence, in every section, is readable for someone with no prior exposure to your project.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    The work is compressed to 1–3 claims, and each one is paired with evidence.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    Likely objections from a skeptical reader are anticipated and addressed in the text.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    The title is descriptive of your findings.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    The TL;DR is a scannable bullet list, stands alone, and names the question, the method, and the answer.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    The threat model and the relevance to AI safety are explicit in the introduction.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    All relevant limitations are included in the Discussion, with nothing uncomfortable omitted.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    Claims are calibrated: what you showed, what you believe, and what you speculate are clearly distinguished.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    Every figure has a caption that stands on its own.
    
- ![unticked](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAYAAABV7bNHAAAA1ElEQVR4Ae3bMQ4BURSFYY2xBuwQ7BIkTGxFRj9Oo9RdkXn5TvL3L19u+2ZmZmZmZhVbpH26pFcaJ9IrndMudb/CWadHGiden1bll9MIzqd79SUd0thY20qga4NA50qgoUGgoRJo/NL/V/N+QIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIECBAgQIAAAQIEyFeEZyXQpUGgUyXQrkGgTSVQl/qGcG5pnkq3Sn0jOMv0k3Vpm05pmNjfsGPalFyOmZmZmdkbSS9cKbtzhxMAAAAASUVORK5CYII=)
    
    The code is linked from the post.
    

## References and further reading

Among other resources, this guide draws directly on:

  

- [Alignment Research Project Blog Post Template - BlueDot Impact](https://docs.google.com/document/d/1-0eeUR0OAYcEOT7kW2Xx5mBz1uSLoErqANhLJOW5r6k/edit?usp=drivesdk)
    
- [Highly Opinionated Advice on How to Write ML Papers - Neel Nanda](https://www.alignmentforum.org/posts/eJGptPbbFPZGLpjsp/highly-opinionated-advice-on-how-to-write-ml-papers)
    
- [The Paper-Writing System I'd Hand Every New PhD Student on Day One - Lennart Nacke (X thread)](https://x.com/acagamic/status/2021617206958735696)
    

  
**