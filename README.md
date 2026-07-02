# Redrob Talent Ranking Proof of Concept

This repository contains a CPU-only proof of concept for the Redrob India Runs Data & AI Challenge. The goal is to rank candidates for a Senior AI / ML-oriented hiring brief by combining structured profile data, career history, education, and platform engagement signals.

## What the ranker does

The implementation in [rank.py](rank.py) scores each candidate across these dimensions:

- Skill match against JD-relevant capabilities such as embeddings, retrieval, vector search, ranking, and Python.
- Career relevance, with preference for product-company and ML/AI experience over consulting-only backgrounds.
- Experience fit, centered on the 5-9 year band called out in the brief.
- Location fit, with preference for the India cities highlighted in the challenge prompt.
- Behavioral and availability signals such as recent activity, open-to-work status, response rate, notice period, profile completeness, interview completion, verification, and visibility.
- GitHub activity and education as smaller tie-break signals.

It also includes a conservative honeypot check to exclude obviously inconsistent profiles.

## Data format note

The provided file named `candidates.jsonl` is not line-delimited JSON in this workspace. It is a pretty-printed JSON array, and the loader in [rank.py](rank.py) is written to handle both array-style JSON and true JSONL input.

## Reproduce the submission

Run the ranker from the repository root:

```powershell
py -3 rank.py --candidates ".\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl" --out .\submission.csv
```

Then validate the output with the challenge checker:

```powershell
py -3 .\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\validate_submission.py .\submission.csv
```

## Output format

The submission file must contain exactly these columns in this order:

`candidate_id,rank,score,reasoning`

It must contain exactly 100 ranked rows, with unique candidate IDs and ranks from 1 to 100.

## Notes

- The implementation is dependency-light and intended to run within the challenge constraints.
- The reasoning field is generated from the candidate record itself and avoids unsupported claims.