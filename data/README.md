# Data

`raw/` contains the course-provided GPT loss and learning-rate curves.

Main file:

```text
data/raw/gpt_loss+lrs.pkl
```

It is a pickle containing three schedules:

```text
M:100M_gpt_D:20B_scheduler:811_rope
M:100M_gpt_D:20B_scheduler:wsd_rope
M:100M_gpt_D:20B_scheduler:cosine_rope
```

Each curve has `step`, `Metrics/loss`, and `lr`. `data/raw/pkl_to_csv.py` reindexes the curves to the dense step grid and forward-fills the two isolated missing rows documented in the presentation.
