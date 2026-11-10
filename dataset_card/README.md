---
license: cc-by-4.0
pretty_name: "blp-research: synthetic business liaison notes, regional indicators and a drafting-assistant experiment"
language:
  - th
  - en
size_categories:
  - 10K<n<100K
task_categories:
  - text-classification
  - time-series-forecasting
tags:
  - business-liaison
  - central-banking
  - text-as-data
  - nowcasting
  - regional-economics
  - thai
  - retrieval-augmented-generation
  - randomized-experiment
  - synthetic
configs:
  - config_name: notes
    data_files: data/notes.csv
    default: true
  - config_name: gold_labels
    data_files: data/gold_labels.csv
  - config_name: regional_activity
    data_files: data/regional_activity.csv
  - config_name: human_coders
    data_files: data/human_coders.csv
  - config_name: experiment_simulated
    data_files: data/experiment_simulated.csv
  - config_name: indicators
    data_files: results/indicators.csv
  - config_name: nowcast_predictions
    data_files: results/nowcast_predictions.csv
---

# blp-research: synthetic business liaison notes, regional indicators and a drafting-assistant experiment

> **This dataset is entirely synthetic.** The Bank of Thailand's Business Liaison Program (BLP) notes are
> confidential and none of them are here. Every note, label, series and result comes from the project's data
> generator and exercises the software; none of it says anything about Thai firms or regions. Firm names, contact
> names and telephone numbers in the generator are fictitious, and the notes are published only after masking.

Data for the preprint *From Liaison Notes to Regional Indicators: Text-Based Measurement of Regional Economic
Activity from Business Liaison Program Data and a Randomized Evaluation of a Generative-AI Report-Drafting
Assistant* by Thanatchon Chongmankhong ([read online](https://thanatchon36.github.io/blp-research/paper/) ·
[PDF](https://github.com/thanatchon36/blp-research/blob/main/paper/main.pdf)).
The paper converts Thai-language liaison interview notes into sector-weighted, composition-adjusted regional
diffusion indices, tests them in pseudo-real-time nowcasts, builds a retrieval-augmented assistant that cites a
source for every sentence and blocks numbers it cannot verify, and designs a randomized crossover experiment on
report-preparation time. Code: <https://github.com/thanatchon36/blp-research>;
archived on Zenodo: [{DOI}](https://doi.org/{DOI}).

## Files

| path | content |
|---|---|
| `data/notes.csv` | 7,560 synthetic interview notes from 1,714 firms in three regions (North, Northeast, South), 13 provinces and five sectors, 2012Q1 to 2025Q4. `text_anon` is the Thai note after masking: firm, contact and telephone become `[บริษัท]`, `[บุคคล]`, `[โทรศัพท์]` |
| `data/gold_labels.csv` | the generator's label for each note and topic (sales, employment, investment, price, cost, outlook): 1 up, 0 flat, -1 down, empty = not mentioned (45,360 rows) |
| `data/regional_activity.csv` | the target series by region and quarter: `y`, the official-style growth rate (% y-o-y), and `activity`, the latent activity the notes are drawn from (168 rows) |
| `data/sector_weights.csv` | illustrative output shares by region and sector used to weight the indices |
| `data/human_coders.csv` | two simulated coders on 400 notes (2,400 note-topic rows), for the inter-coder reliability demonstration |
| `data/experiment_simulated.csv` | a simulated run of the Study 4 crossover experiment: 30 participants, four tasks each, with condition (AI or control), minutes, blinded quality score, words and errors; the true time reduction is set to 30% |
| `results/scoring_reliability.csv` | lexicon coder against the gold labels: accuracy, macro-F1 and Cohen's kappa by topic |
| `results/indicators.csv` | regional indices by quarter: raw, sector-weighted (`_w`) and composition-adjusted (`_adj`) diffusion indices per topic, the activity index `BLP_RAI`, outlook, price, cost and uncertainty indices, and the number of notes |
| `results/nowcast_predictions.csv`, `results/nowcast_eval.csv` | expanding-window nowcasts from 2018Q1 for the AR benchmark, AR + BLP and AR + BLP + outlook, with RMSE, MAE, Clark–West and Diebold–Mariano tests |
| `results/draft_report.md`, `results/verification.json` | a regional report drafted by the extractive baseline (Thai) and the verifier's check of every sentence, including a negative control with injected errors |
| `results/power_curve.csv`, `results/power_grid.csv`, `results/experiment_dryrun.json` | power simulations for Study 4 and the pre-registered analysis applied to the simulated experiment |
| `results/numbers.json`, `results/summary.json` | every number quoted in the paper, and the pipeline's run summary |
| `SHA256SUMS` | checksums of every file above |

## The synthetic data

Three regional factors load on a national AR(1) cycle with a COVID-19 shock in 2020Q2 (tourism recovers more slowly
than other sectors). Each quarter about 45 firms per region are interviewed, 80% of them retained from the previous
quarter; each firm has its own optimism, and firms that join later are more optimistic, which biases a raw diffusion
index and is what the composition adjustment removes. Notes are assembled from Thai phrase templates by topic and
direction, with mixed statements (an increase and a decrease in one topic, labelled flat) and distractor sentences
about competitors, the government or world markets that a lexicon misreads. The outlook topic carries part of next
quarter's activity. The sample's sector mix is deliberately unlike the output weights. Seed 42. Because the notes
come from templates, the lexicon's agreement with the gold labels is an upper bound for real notes.

`synthetic.generate(seed=42)` in the code regenerates everything here, including the unmasked text;
`make dataset` rebuilds this folder.

## Licence

CC BY 4.0 (`LICENSE`).

## Citation

```bibtex
@misc{chongmankhong2026liaison,
  author = {Chongmankhong, Thanatchon},
  title  = {From Liaison Notes to Regional Indicators: Text-Based Measurement of Regional Economic Activity
            from Business Liaison Program Data and a Randomized Evaluation of a Generative-AI
            Report-Drafting Assistant},
  year   = {2026},
  doi    = {{DOI}},
  url    = {https://thanatchon36.github.io/blp-research/paper/},
  note   = {Preprint; results are from synthetic data}
}
```
