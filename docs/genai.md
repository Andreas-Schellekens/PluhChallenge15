# GenAI use

How generative AI was used in this project, what the team decided and checked, and where
the limits were. Drafted by Claude Code; completed by the team (team contributions and the statement).

## Tools

| tool | used for |
|---|---|
| **Claude Code** (Anthropic, AI coding assistant) | writing and debugging the scripts and the app; proposing puck positions for labelling; running training and evaluation; drafting the documentation and these exam notes |
| **Roboflow** (labelling platform) | storing images, reviewing labels (Review queue), dataset versions and exports |
| Roboflow auto-label (Gemini, SAM 3, GPT) | tried once on the 4K clip and **not used**: the boxes were wrong or missing (see `annotation.md`, pilot) |

## Who did what

| part | what the AI did | what the team decided or checked |
|---|---|---|
| Scope and data | – | chose ice hockey; switched to `fillempje.mp4` (renamed `hockey_static_1080p.mp4`) because the 4K clip was handheld and showed few pucks. These choices were made in team meetings with everyone present |
| Baseline (E0, E0-B) | wrote `baseline.py`, ran it, summarised the numbers | re-run by the team members with a GPU (the GPU is noted in every log entry) |
| Labelling (rounds 1 and 2) | proposed candidates: background subtraction (`propose_puck.py`), later our own model (`propose_model.py`); sorted them into confident / unsure / wrong; uploaded them as **predictions to Roboflow Review** | **Andreas reviewed every label**: approved, rejected, moved or drew boxes, and checked 24 frames by hand (23 marked null). No AI label entered the dataset without human approval |
| Labelling rules | drafted the rules (box size, ≥ 50% visible, null images, time split) | discussed in team meetings; the blade rule was decided by Andreas |
| Fine-tuning (E1, E2a, E2b) | wrote `make_puck_dataset.py`, `train_puck.py`, `eval_puck.py`; chose the tiling, augmentations and settings and explained why; ran training and evaluation | chose to do a second labelling round |
| Homography (H1, H2) | wrote `homography.py` and `rink.py`, measured the keypoints, fitted the lens correction | visual check of the overlay |
| Tracking (T1) | wrote `kalman.py`, `track.py` and the tuned ByteTrack config; designed the gap test | chose tracking as the Extra Mile, after options were explained |
| App | wrote `app.py` and `pipeline.py`, tested them in the browser | run and tried by the team members with a GPU |
| Error analysis | checked samples visually and wrote the analyses in `experiments.md` | – |
| Audit (A1) | a second Claude Code agent read all scripts and compared every number in the docs with the saved outputs; the main session verified the findings and fixed them (homography, gap test, dataset random stream, lens mask, app) | Finn asked for the full check and gave the go-ahead |
| Homography (H3) | found that the far goal landed at about 100 m, measured two more keypoints, changed the fit to pixels | the new overlay is not yet checked by a team member |
| Team colours (TM1) | wrote `teams.py`; found and fixed the failed first version (features not scaled); **labelled the 120-crop sample itself** | chose team colours as the second Extra Mile; **Andreas labelled the 120-crop sample again, blind** (TM2): 90–92% instead of Claude's 95.5% |
| Seed study (S1) | wrote `seed_summary.py`, ran the six training runs, wrote the analysis | the false positives of the seed runs were only counted, not looked at by a team member |
| Local setup on Mihai's PC | rebuilt the puck tiles from the Roboflow v3 export, unpacked the team's outputs, re-validated the E2a weights (0.635 / 0.382, same as logged); found that the Quadro T1000 (4 GB) cannot train the puck model in reasonable time (about 45 h per run) | Mihai asked for it and supplied the team's export; the E0 outputs of his original runs (his PC) confirm the `yolo26s_640` row and the Quadro fps that A1 listed as unverified |
| Lens calibration (H4) | wrote `calibrate_lens.py` (plumb-line scan over k1 with an exact inverse of the lens model), ran it, wrote the H4 entry; found that `Lens.undistort` (5 OpenCV iterations) is up to ~2 px off near the image edge; set the app's Tracking and Puck model tabs to the current runs | Mihai asked for it, after A1 listed the H2 k1 table as unreproducible; `mask.jpg` and `straightened.jpg` are not checked by a team member; `Lens.undistort` was left as it is |
| Conf/NMS sweep and error analysis (P1) | wrote `sweep_conf_nms.py`, `docs/error_analysis.md` and the P1 entry; ran the sweep; typed the five E2a errors first from the crops. The error types quoted from other runs (E1, E2b) are Claude's visual checks, not yet confirmed by a team member | **Zjef checked the five E2a error types on enlarged crops**; the conclusion that the val "puck" at f5070 is a stick blade is his. Andreas removed f5070 in Roboflow (v4, E3). f4950 stays unclear |
| Consistency check, round 1 | wrote `label_agreement.py` and `label_blind.py`; described what the disagreement frames show (not yet checked by Andreas) | **Zjef labelled the 10 frames blind**; his reasons for the disagreements are his own answers. Afterwards Finn decided the ±10 frames check and the box margin (next row) and Andreas the blade rule |
| Labelling decisions after round 1 | added the ±10 frames view (`--video`, key `v`) and the automatic 2 px margin (`--margin`) to `label_blind.py`; wrote the decisions and the round-2 commands into `annotation.md`; tested on a synthetic video, including a simulated labelling session with scripted keys and clicks (not yet by a person with the real clip) | **Finn decided**: 2 px margin and ±10 frames check adopted, no blade rule for now (Andreas decided it later the same day), a second round agreed |
| Roboflow v4, retraining and final report (E3) | generated and exported Roboflow v4 through the Roboflow connector; rebuilt the tiles, re-scored E1/E2a/E2b, retrained the E2a recipe over 3 seeds; wrote the E3 entry and the draft `docs/final_report.md`; took the app screenshots with headless Chrome and rendered the training console screenshot from the saved log | Andreas asked for v4 and the report, decided the blade rule and removed f5070 in Roboflow; the team reads the report before the upload |


## Team contributions

The main choices and agreements (sport and clip, labelling rules, which Extra Mile tracks, who does
what) were made together in team meetings with everyone present. Besides that:

- **Andreas** did all the labelling in Roboflow: he reviewed every AI proposal in both rounds and drew,
  moved or rejected the boxes himself, removed f5070 and generated v4. He labelled the team-colour check sample blind (TM2). He decided the blade rule and put
  together the documentation (annotation rules, this GenAI file, the final report), with Claude Code drafting.
- **Finn** asked for the audit (A1) and decided the 2 px margin and the ±10 frames check.
- **Mihai** ran the original baselines (E0) on his PC and asked for the lens calibration (H4).
- **Zjef** ran the conf/NMS sweep and error analysis (P1) on his PC, checked the five E2a errors (f5070 is
  a stick blade) and labelled the blind consistency round.

## How AI output was verified

- **Labels.** Every AI proposal went through human review in Roboflow. The AI was often
  wrong: only 65% (round 1) and 74% (round 2) of its "confident" proposals were approved, and
  of the "unsure" ones only 8% and 38%. This shows why the review was necessary.
- **Results.**
  - Numbers in `experiments.md` come from a saved run (`outputs/`) with the exact command, so
    they can be reproduced. The audit (A1) checked this and lists the few that cannot be
    confirmed from a saved output.
  - Numbers that rest on a visual check by the AI and not on a label are marked as such (the
    clip samples in E1, E2 and T1). The 110-crop team sample of TM1 was checked by Andreas (TM2) and
    gave a lower score than Claude's labels.
- **AI checking AI.** The audit was done by a second AI agent. It found real
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

## Statement

We used GenAI (Claude Code) in three ways: to help write the code, to understand the assignment better,
and to give us ideas and options. The decisions were ours: which sport and clip, which labels to keep,
the labelling rules, which Extra Mile tracks to do and which model to use. All big choices were
discussed in team meetings with everyone present, and the team members with a GPU ran and checked the
code and the results on their own PC.

Checking the AI was necessary, and it paid off. Examples where the AI's output was wrong or too
optimistic and we caught it:

- only 65–74% of the AI's "confident" puck proposals were real pucks, so every label was reviewed by a person;
- the first homography put the far goal about 70 m too far away (H3);
- the first test of the Kalman filter made it look better than it was (A1, T2);
- a val label (f5070) turned out to be a stick blade, found when we checked the model's errors by hand (P1).

Every team member is expected to explain the whole project on the oral exam, so we go through each part of
the pipeline together before the exam.
