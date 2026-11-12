# GenAI use

How generative AI was used in this project, what the team decided and checked, and where
the limits were. **Draft by Claude Code (2026-11-05, updated 2026-11-11), to be completed and confirmed by the
team.** Check it against the course GenAI policy for any required format or wording.

## Tools

| tool | used for |
|---|---|
| **Claude Code** (Anthropic, AI coding assistant) | writing and debugging the scripts and the app; proposing puck positions for labelling; running training and evaluation; drafting the documentation and these exam notes |
| **Roboflow** (labelling platform) | storing images, reviewing labels (Review queue), dataset versions and exports |
| Roboflow auto-label (Gemini, SAM 3, GPT) | tried once on the 4K clip and **not used**: the boxes were wrong or missing (see `annotation.md`, pilot) |

## Who did what

| part | what the AI did | what the team decided or checked |
|---|---|---|
| Scope and data | – | chose ice hockey; switched to `fillempje.mp4` because the 4K clip was handheld and showed few pucks; processed the teacher feedback (fine-tune + comparison, working homography, frontend with parameters) |
| Baseline (E0, E0-B) | wrote `baseline.py`, ran it, summarised the numbers | – |
| Labelling (rounds 1 and 2) | proposed candidates: background subtraction (`propose_puck.py`), later our own model (`propose_model.py`); sorted them into confident / unsure / wrong; uploaded them as **predictions to Roboflow Review** | **Andreas reviewed every label**: approved, rejected, moved or drew boxes, and checked 24 frames by hand (23 marked null). No AI label entered the dataset without human approval |
| Labelling rules | drafted the rules (box size, ≥ 50% visible, null images, time split) | to be confirmed by the team; open question: does a puck touching the stick blade count |
| Fine-tuning (E1, E2a, E2b) | wrote `make_puck_dataset.py`, `train_puck.py`, `eval_puck.py`; chose the tiling, augmentations and settings and explained why; ran training and evaluation | chose to do a second labelling round |
| Homography (H1, H2) | wrote `homography.py` and `rink.py`, measured the keypoints, fitted the lens correction | visual check of the overlay |
| Tracking (T1) | wrote `kalman.py`, `track.py` and the tuned ByteTrack config; designed the gap test | chose tracking as the Extra Mile, after options were explained |
| App | wrote `app.py` and `pipeline.py`, tested them in the browser | – |
| Error analysis | checked samples visually and wrote the analyses in `experiments.md` | – |
| Audit (A1, 2026-11-11) | a second Claude Code agent read all scripts and compared every number in the docs with the saved outputs; the main session verified the findings and fixed them (homography, gap test, dataset random stream, lens mask, app) | Finn asked for the full check and gave the go-ahead |
| Homography (H3) | found that the far goal landed at about 100 m, measured two more keypoints, changed the fit to pixels | to do: visual check of the new overlay |
| Team colours (TM1) | wrote `teams.py`; found and fixed the failed first version (features not scaled); **labelled the 120-crop sample itself** | chose team colours as the second Extra Mile; to do: confirm the sample labels |
| Seed study (S1) | wrote `seed_summary.py`, ran the six training runs, wrote the analysis | to do: look at the false positives of the seed runs (only counted so far) |

*Team: complete this table with your own contributions (for example Mihai's part, extra
checks you did yourselves, and decisions made in team meetings).*

## How AI output was verified

- **Labels.** Every AI proposal went through human review in Roboflow. The AI was often
  wrong: only 65% (round 1) and 74% (round 2) of its "confident" proposals were approved, and
  of the "unsure" ones only 8% and 38%. This shows why the review was necessary.
- **Results.**
  - Numbers in `experiments.md` come from a saved run (`outputs/`) with the exact command, so
    they can be reproduced. The audit (A1) checked this and lists the few that cannot be
    confirmed from a saved output.
  - Numbers that rest on a visual check by the AI and not on a label are marked as such (the
    clip samples in E1, E2 and T1, and the 110-crop team sample in TM1).
- **AI checking AI.** The audit of 2026-11-11 was done by a second AI agent. It found real
  errors in earlier AI work (a homography that was 70 m off at the far end, an unfair test of
  the Kalman filter, a confounded comparison). That shows two things: AI output that looks
  finished can be wrong, and an independent check finds such errors. It does not replace a
  check by a person.
- **Code.** The scripts are short, with a docstring that explains the goal, outputs and usage.
  Non-obvious choices are explained in comments (tiling, gating, the time split, the sign of
  the homography).

## Limits and risks we saw

- **Visual checks by the AI** (clip samples, grouping proposals) are not ground truth. A person
  should confirm them before they count as a result.
- **Explanation.** A lot of code was generated in a short time. Every team member must be able
  to explain it without AI on the oral exam. For that we go through the parts together.
- **Data.** No personal data was sent to AI services beyond the match video frames. Videos,
  datasets and weights are not in git.

## Statement (to be completed by the team)

*Describe here in your own words how you used GenAI, what you learned from it, and which parts
you can defend yourselves.*
