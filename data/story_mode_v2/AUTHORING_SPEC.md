# Authoring spec — narrativity pairs v2 (scale-up to 1000)

You are writing contrastive text pairs for a mechanistic-interpretability experiment that extracts
the **narrativity/fictionality direction** from a language model's residual stream. Each pair is one
short **invented narrative** and one **non-narrative** text on the same topic, matched for length.
There is no task, no question, no instruction, no request anywhere in either text.

The diff-in-means over these pairs *is* the vector. Anything that correlates with the narrative arm
across the whole dataset becomes part of the vector. That is why the constraints below are hard.

---

## 1. The one rule that matters most: predicate type

Narrativity is **individuated events ordered in time**. The non-narrative arm must contain none.

Deleting the character or switching to habitual aspect **does not** make a text non-narrative.
"A balloon rose sideways and the cord came taut against the mast" is still a temporally-ordered
change of state with the participant deleted. It is a narrative.

| BANNED in the non-narrative arm | REQUIRED instead |
|---|---|
| change-of-state verbs — rose, fell, broke, formed, became, began, stopped, grew, lost | copular and relational — is, was, differs from, consists of, depends on, corresponds to |
| motion verbs with a path — carried out, came taut, falls away, reaches, arrives, passes through | stative-locational — lies, stands, sits, is situated |
| perception or cognition with an animate subject — she saw, they noticed, he felt | classificatory — counts as, falls into, is defined by, is graded on |
| temporal connectives — then, when, once, before, after, until, still, already, meanwhile, finally | deontic — is required, is permitted, is to be |
| completive aspect, one-off outcomes — "two bars had folded inward" | quantificational — mean, percent, of the order of, in nine cases of ten |

**Three failure modes to check every non-narrative text against:**

1. **Habitual narrative.** "Every morning crews would walk down to the outfall" — iterative aspect,
   but still a scene with participants. Strip the participants, not just the tense.
2. **Overshooting into abstraction.** "Protocols specify verification of structural integrity at
   defined intervals" — loses the shared vocabulary, so the pair reverts to a trivially easy
   contrast and stops testing anything.
3. **Drifting into procedure.** "First, work a glove under the lip. Then check the bolt seats." A
   procedure is ordered events in time with a generic participant — a generic narrative. There is
   **no procedural style in this dataset**. State what *is the case*, never what *to do*.

Mild functional predicates are tolerated where a mechanism is being described ("a breastshot wheel
takes water at axle height and converts the weight of the flow") because they are generic-functional
rather than event-sequential. Do not use this as licence for narration.

---

## 2. Hard constraints — both arms

| | |
|---|---|
| **Length** | 60–90 words each. Within a pair, \|Δ\| ≤ 3 words. Count with `[A-Za-z0-9][A-Za-z0-9'\-]*` — hyphenated forms are ONE word. |
| **Person** | Third person only. **Zero occurrences** of i, me, my, mine, we, us, our, you, your, yours — as whole words, anywhere, including inside quoted speech. This rules out dialogue and letters. |
| **Ending** | Both arms end on a full stop, mid-text, clearly continuable. No dangling colon, no trailing dash. |
| **Topic** | Matched within the pair — the two texts are about the same subject matter. |
| **No self-labelling** | The narrative never says "in this story"; the non-narrative never says "this is not a narrative". |
| **No harm content** | Nothing violent, sexual, criminal, medical-advisory or otherwise harmful in either arm. Deaths may be referred to obliquely if a genre needs it, never depicted. |
| **No evaluation vocabulary** | Avoid test, assess, evaluate, benchmark, audit, trial, score-as-verb. These belong to a rival direction being measured against this one. `sample`, `median`, `percent` are permitted in `statistical` only. |
| **No named real people, places or works** | Invented proper nouns only. Generic real geography (a river, a northern coast) is fine; "Amsterdam", "Beethoven", "the Thames" are not. |
| **ASCII only** | No curly quotes, no ellipsis character. An em dash `—` is permitted in the narrative arm only. |

## 3. Hard constraints — narrative arm

- **Invented.** Every narrative is fiction. Not a real event, not a report of one.
- **Individuated.** One specific occasion, specific participants, events in sequence. Give the
  central figure an invented given name or a definite role ("the youngest cooper").
- **No second-person address, no dialogue.** Interiority is allowed (`free_indirect` mode).

## 4. Hard constraints — non-narrative arm

- Built from the allowed predicate types in §1.
- Non-assertive about contested empirical facts — prefer definitions, structures, criteria,
  taxonomies, requirements over verifiable claims about the world. A negative pole full of
  checkable true statements turns the axis into a truth detector.
- No proper nouns at all except month names inside a seasonal statement.

---

## 5. The 16 non-narrative styles

Each has a **dominant construction**. Stay inside it — this is what keeps the styles from collapsing
into one template.

| style | what it does | dominant construction |
|---|---|---|
| `criterial` | the categories and thresholds by which outcomes in a domain are classified — **not** the outcomes happening | "X is graded on A and not on B. C is disqualifying whatever its D. E bears on the cause and never on the class." |
| `technical` | the structure and relations of a device or system | "A is carried on B. Its P is a function of Q and of R. Without S the two are independent." |
| `taxonomy` | how two or three classes are distinguished | "A differs from B in what it removes rather than what it adds. C is a third class, distinguished from both by D." |
| `topographic` | stative spatial description of a place — setting with no events | "Three bars lie parallel to the shingle. The depth at slack is under two feet. The landward side is the drier." |
| `statistical` | survey or report prose — people present, no events | "P varies more within Q than between them. The median is R. S accounts for most of the spread." |
| `encyclopedic` | definitional account of a kind of thing | "An A is a B characterised by C. Its D depends on E. F is therefore narrow and fixed by G." |
| `spec` | requirements and permissions | "A was required to be B. C was permitted only where D. E was to bear on three supports." |
| `essay` | argumentative or analytic prose | claim, then the reason, then a concession that does not concede the point |
| `glossary` | a short series of term-and-definition entries | "A: the interval between B and C. D: any B outside that interval. E: the same, at the margin." |
| `legal_def` | statutory definition and condition | "For the purposes of this part, A means B. Where C obtains, D applies. Nothing in E affects F." |
| `rules` | constraints on legal positions or moves — **not** ordered steps | "A move is legal only if A. A position in which B is drawn. C is not a move and cannot be answered." |
| `catalogue` | what a collection or range consists of, concretely and statively | "The range consists of three sizes, of which the middle is the commonest. Each is D in E, with F at the G." |
| `criticism` | evaluative claims about a class of works or practices | "The weakness of the form is A. What passes for B in it is usually C. Its best examples are the ones that D." |
| `conceptual` | philosophical or definitional argument about a concept | "The question is whether A is a species of B or a different thing under one name. If A, then C." |
| `quantitative` | relations between quantities | "P is proportional to the square of Q. The constant is a property of R. Beyond S the relation does not hold." |
| `morphology` | the parts of a thing and how they stand to each other | "The A consists of a B and two C. The B is attached at D. E is absent in the smaller forms." |

### `criterial` is the hard rung — extra requirement

`criterial` pairs must **share at least 8 content-word types with their narrative twin** (nouns,
mostly). Write the narrative first, then write the criterial text about *how outcomes in that domain
are classified*, reusing its object vocabulary. Length, register and concreteness are all matched
there, so a probe that separates a `criterial` pair must be reading narrativity itself. Example:

> Teodora carried the balloon out at ten to six, both arms under it, and the wind took the skin
> against her chin before she reached the launch square. The sonde swung on its cord and knocked her
> knee twice. She waited for the gust to pass, counted, and let go a little too early — the balloon
> went up sideways and the cord snapped taut against the mast.

> A failed launch is a matter of ascent angle, not of damage. Sideways ascent, meaning departure
> from vertical beyond thirty degrees, is one class of failure; contact between the cord and the mast
> is a second, independent of the first. Neither concerns the sonde. Wind speed at the square is the
> usual cause of both, and the gust interval, not the balloon or the skin, sets the margin.

---

## 6. The 8 narrative modes

| mode | delivery |
|---|---|
| `scenic` | close third-person scene, one continuous occasion |
| `oral` | anecdotal, informal register, as a community would retell it ("Nobody in the shed would work behind Tomos, and there was a reason for it.") |
| `folktale` | fable or legend register ("There was a miller who owed the river a debt...") |
| `chronicle` | annalistic, dated, one entry's worth of events ("In the ninth year of the works...") |
| `free_indirect` | narration carrying the figure's thought without quoting it |
| `reported` | flat, journalistic narration of one invented specific event |
| `mythic` | cosmological or origin register, large scale, still specific |
| `episodic` | a run of occasions with one figure, each briefly rendered |

`folktale`, `mythic` and `chronicle` stay in past tense. The other five may be past or present.

## 7. The 24 genre tags

**realistic:** `realist_domestic` `realist_work` `historical` `premodern` `mystery` `noir`
`expedition` `sea_voyage` `western` `war` `institutional` `sport_performance`

**non_realistic:** `ghost` `weird` `horror_quiet` `science_fiction` `space` `post_apocalyptic`
`high_fantasy` `folk_fantasy` `myth` `animal_fable` `magical_realism` `alt_history`

`realism` is determined by which list the genre is in.

---

## 8. Anti-repetition — read this twice

Twenty-five authors are writing this dataset from disjoint topic domains. The failure that would
ruin it is **template convergence**: every `taxonomy` text opening "X differs from Y in...", every
`criterial` text opening "A is graded on...". A probe trained on that measures template recall, not
narrativity.

Within your own batch of 40:

- **No two texts may share a content-bearing 5-gram** (five consecutive words ignoring
  the/a/of/and/is/was...). Check your own output before writing the file.
- **Vary the opening construction.** Across your 40 non-narrative texts, at most **two** may begin
  with the same first content word. Use the dominant construction's *shape*, not its exact wording —
  invert it, lead with the exception, lead with the quantity, lead with the negative case.
- **Do not reuse the opening words of the §5 examples.** The first batches written against this spec
  all opened their `legal_def` texts on "For the purposes of...", their `conceptual` texts on "The
  question is whether...", their `criticism` texts on "The weakness of the form is...", and their
  `catalogue` texts on "The range consists of...". The examples show the *shape* of each style; 25
  authors copying their first three words is exactly the convergence this section forbids. Treat
  these openers as banned: **purposes, question, weakness, range, move, differs, consists**. Reach
  the same construction from a different entry point — start from the exception, the boundary case,
  the quantity, the thing excluded, the second of two classes.
- **Vary sentence count.** Some texts 3 sentences, some 5, some 6. Not all four.
- **Vary narrative openings.** Do not open more than three narratives with a bare name as the first
  word. Open with a place, a time, an object, a condition, a negation, the middle of a situation.
- Use **at least 12 of the 24 genre tags** and at least 35 distinct topics among your 40 pairs.

---

## 9. Your quotas — exactly 40 pairs

| dimension | quota |
|---|---|
| `nonnarr_style` | `criterial` ×5; then ×2 of each of the other 15 styles (30); then ×1 of each of the five styles named as YOUR EXTRAS (5) |
| `narr_mode` | ×5 of each of the 8 modes |
| `realism` | 20 realistic / 20 non_realistic |
| `tense_polarity` | `default` ×16 (past narrative + present non-narrative) · `reversed` ×10 (present narrative + past non-narrative) · `both_past` ×10 · `both_present` ×4 |

Present-tense narratives (the 10 `reversed` + 4 `both_present` = 14) must be `scenic`,
`free_indirect`, `reported`, `episodic` or `oral`.

## 10. Output format

Write **one JSON object per line**, no wrapping array, UTF-8, to the path you are given. Fields, all
required, exactly these names:

```json
{"pair_id": "b07-01", "batch": 7, "domain": "music_and_sound", "split": "train", "narr_mode": "scenic", "genre": "premodern", "realism": "realistic", "nonnarr_style": "criterial", "tense_narr": "past", "tense_nonnarr": "present", "tense_polarity": "default", "context": "bell_foundry_flaw", "text_narrative": "...", "text_nonnarrative": "..."}
```

- `pair_id`: `b<NN>-<kk>` with your batch number zero-padded and kk from 01 to 40.
- `context`: a snake_case slug naming the specific topic. **Must be unique within your batch** and
  specific enough that no other author would coin it (`bell_foundry_flaw`, not `metal`).
- `split`: the value you are given, on every line.

## 11. Before you write the file

Run these checks on your own 40 pairs and fix what fails:

1. Every text 60–90 words; every pair \|Δ\| ≤ 3.
2. Zero first/second-person pronouns, both arms.
3. Every text ends with a full stop.
4. Every non-narrative text checked against the three failure modes in §1.
5. Every `criterial` pair shares ≥ 8 content words with its twin.
6. All 40 `context` slugs distinct; quotas in §9 exactly met.
7. No content 5-gram shared between any two of your 80 texts.

Then write the file. Report back only: the path written, the achieved quota tallies, and anything
you could not satisfy.
