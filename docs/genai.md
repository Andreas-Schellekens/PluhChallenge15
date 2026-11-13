# GenAI use

How generative AI was used in this project, what the team decided and checked, and where
the limits were. **Draft by Claude Code (2026-11-05, updated 2026-11-11 and 2026-11-13), to be completed and confirmed by the
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
| Scope and data | – | chose ice hockey; switched to `fillempje.mp4` (renamed `hockey_static_1080p.mp4` on 2026-11-13) because the 4K clip was handheld and showed few pucks; processed the teacher feedback (fine-tune + comparison, working homography, frontend with parameters) |
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
| Local setup on Mihai's PC (2026-11-13) | rebuilt the puck tiles from the Roboflow v3 export, unpacked the team's outputs, re-validated the E2a weights (0.635 / 0.382, same as logged); found that the Quadro T1000 (4 GB) cannot train the puck model in reasonable time (about 45 h per run) | Mihai asked for it and supplied the team's export; the E0 outputs of his original runs (2026-10-16, his PC) confirm the `yolo26s_640` row and the Quadro fps that A1 listed as unverified |
| Lens calibration (H4, 2026-11-13) | wrote `calibrate_lens.py` (plumb-line scan over k1 with an exact inverse of the lens model), ran it, wrote the H4 entry; found that `Lens.undistort` (5 OpenCV iterations) is up to ~2 px off near the image edge; set the app's Tracking and Puck model tabs to the current runs | Mihai asked for it, after A1 listed the H2 k1 table as unreproducible; to do: visual check of `mask.jpg` and `straightened.jpg`; the team decides whether to fix `Lens.undistort` (it would change T2/TM1 slightly) |
| Conf/NMS sweep and error analysis (P1, 2026-11-13) | wrote `sweep_conf_nms.py`, `docs/error_analysis.md` and the P1 entry; ran the sweep; typed the five E2a errors first from the crops. The error types quoted from other runs (E1, E2b) are Claude's visual checks, not yet confirmed by a team member | **Zjef checked the five E2a error types on enlarged crops**; the conclusion that the val "puck" at f5070 is a stick blade is his. To do: fix the f5070 label in Roboflow (v4); a second person looks at f4950 |
| Consistency check, round 1 (2026-11-13) | wrote `label_agreement.py` and `label_blind.py`; described what the disagreement frames show (not yet checked by Andreas) | **Zjef labelled the 10 frames blind**; his reasons for the disagreements are his own answers. The proposed rules (blade rule, ±10 frames check, box margin) were decided afterwards by Finn, see the next row |
| Labelling decisions after round 1 (2026-11-13) | added the ±10 frames view (`--video`, key `v`) and the automatic 2 px margin (`--margin`) to `label_blind.py`; wrote the decisions and the round-2 commands into `annotation.md`; tested on a synthetic video, including a simulated labelling session with scripted keys and clicks (not yet by a person with the real clip) | **Finn decided**: 2 px margin and ±10 frames check adopted, no blade rule for now, a second round agreed; Finn fixed f5070 in Roboflow |
| Roboflow v4, retraining and final report (E3, 2026-11-13) | generated and exported Roboflow v4 through the Roboflow connector; rebuilt the tiles, re-scored E1/E2a/E2b, retrained the E2a recipe over 3 seeds; wrote the E3 entry and the draft `docs/final_report.md`; took the app screenshots with headless Chrome and rendered the training console screenshot from the saved log | Andreas asked for v4 and the report and decided the blade rule; Finn removed f5070 in Roboflow. To do: the team reads and completes the report |

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
