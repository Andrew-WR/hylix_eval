# Experiment card

## Claim being evaluated

This kit performs a fresh paired behavioral evaluation of an unmodified model
and Hylix. It does not expose or reproduce Hylix's probe-training procedure.

## Pilot setup

- Model: `Qwen/Qwen3.5-9B`, reasoning enabled, 4-bit NF4 inference.
- Held-out tasks: 10 MATH-500 and 10 HumanEval tasks.
- Sampling: temperature 0.6, top-p 0.95, top-k 20.
- Maximum output: 8192 tokens.
- Conditions: normal and Hylix, paired by prompt and random seed.
- Math scoring: normalized exact match on the post-thinking `FINAL:` answer.
- Code scoring: HumanEval tests plus completion-format and normalization audits.

## Historical pilot result

The historical result is recorded in `reported_results.csv`. Overall normalized
accuracy was 35% for normal inference and 50% for Hylix; mean output length was
6677.1 and 6250.6 tokens respectively.

These estimates each use only 20 tasks. They should be treated as exploratory,
not as a precise effect-size estimate.

## Reproduction qualification

The public endpoint permits an independent behavioral replication. It is not a
bitwise replay of the original generations: sampling is stochastic, a fresh
paired seed schedule is used, and the endpoint metadata should be saved with
every run. Compare the new summary with `reported_results.csv`, but do not
expect every individual completion to match the historical one.

The API reports a probe fingerprint. Record it so results from different Hylix
deployments are not accidentally combined.
