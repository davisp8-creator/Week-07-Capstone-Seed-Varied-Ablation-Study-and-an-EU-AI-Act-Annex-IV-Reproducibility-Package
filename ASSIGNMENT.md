# Week-07_Reproducibility-Package

Preserved verbatim from the course page, before the project-specific documentation was written.

## Reproducibility Package
Week 07 | assessment | CLO4 CLO5 CLO6

### Scenario
You are an ML Research Engineer at Cobalt Grid, a scale-up selling short-horizon electricity-demand forecasting to rural utility co-ops in the EU and the US. Head of Research Dr. Ingrid Solberg will not approve the next model release until one narrow architectural question has a defensible answer, and compliance lead Marcus Adeyemi needs technical documentation ready for a conformity file. Choose ONE factor carried forward from earlier weeks — LoRA rank, positional-encoding choice, diffusion or flow-matching noise schedule, quantization precision, or multimodal fusion strategy — and design a controlled ablation across at least three settings with at least three random seeds per setting, all inside a single free-tier Colab GPU session. Build your dataset yourself: either generate synthetic demand traces with Claude Code or Python, or pull an open series from data.gov, an open Kaggle dataset, or an open Hugging Face dataset, and script the acquisition so it reruns cleanly. Report means with seed-level spread, state plainly whether the observed differences exceed run-to-run noise, and log wall-clock latency and peak memory alongside accuracy so the comparison is honest about cost. Package everything for reproduction: pinned dependencies, a seeded config-driven entry point, results tables and plots, and a README a stranger can follow without contacting you. Then write an Annex IV-style technical documentation summary covering system purpose, data provenance, training methodology, evaluation metrics, limitations, and known risks, cross-referencing NIST AI RMF and ISO/IEC 42001 where your controls map. Finally, walk Dr. Solberg and Marcus through the study, the package, and what the study cannot conclude, with your face on camera throughout.

### Requirements
- Camera-on video walkthrough (8-10 min) + project deliverables: reproducible repo (pinned environment, seeded entry-point scripts, config files, results tables and plots, README) and a 3-5 page EU AI Act Annex IV-style technical documentation summary

### Video Submission Requirements
- Maximum duration: 10 minutes
- Format: MP4 (preferred), MOV, or WebM
- Camera: Your face must be visible on camera for the entire video. Picture-in-picture overlay during screen-share is acceptable, but voice-over slides without a visible face is not permitted.
- Script policy: Explain in your own words. Reading verbatim from a written script is not permitted — a brief outline or speaker notes for structure is fine.
- Audio: Clear narration — use a quiet space and speak toward the microphone
- Upload: Submit via the LMS assignment dropbox

Recording tools (free): Zoom (local recording), Loom, OBS Studio, or your device's built-in screen recorder.

### Submission Details
Points: 150. Assignment Type: Project.

### Grading Rubric: Capstone Ablation Study Rubric

| Criterion | Exemplary (150 pts) | Proficient (128 pts) | Developing (105 pts) | Beginning (75 pts) |
|---|---|---|---|---|
| Ablation Design and Control (CLO4, 30 pts max) | Exactly one variable changes between arms, everything else is held fixed, and the student can defend each control decision (30 pts) | Design is controlled with one confound left unaddressed (26 pts) | Multiple variables change at once, weakening any conclusion (21 pts) | No genuine controlled comparison (15 pts) |
| Seed Variation and Statistical Honesty (CLO4, 30 pts max) | Multiple seeds per arm are run and the spread is reported, including when it swamps the effect the student hoped to show (30 pts) | Seeds are varied and spread is reported with minor gaps (26 pts) | Seeds are varied but the spread is downplayed or omitted (21 pts) | Single run per arm presented as a finding (15 pts) |
| Reproducibility Package Quality (CLO6, 30 pts max) | Environment, seeds, data generation, and run order are captured so another engineer could rerun it end to end (30 pts) | Package is reproducible with one missing detail (26 pts) | Package requires guesswork to rerun (21 pts) | Not reproducible (15 pts) |
| Annex IV Technical Documentation (CLO6, 30 pts max) | Documentation addresses the required Annex IV elements accurately and in language a regulator or auditor could follow (30 pts) | Documentation covers most required elements accurately (26 pts) | Documentation is partial or misstates what Annex IV requires (21 pts) | No usable technical documentation (15 pts) |
| Video Communication (CLO5, 30 pts max) | Camera-on capstone presentation in the student own words that states the finding and its limits with equal clarity (30 pts) | Camera-on presentation is clear with minor pacing issues (26 pts) | Presentation overstates the finding or leans on reading from notes (21 pts) | Camera off, or the presentation is read verbatim from a script (15 pts) |

Total Points: 150
