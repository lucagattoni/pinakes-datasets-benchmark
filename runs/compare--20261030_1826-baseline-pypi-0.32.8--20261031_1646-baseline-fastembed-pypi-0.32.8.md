# `20261030_1826-baseline-pypi-0.32.8` (A) against `20261031_1646-baseline-fastembed-pypi-0.32.8` (B)

|  | A | B |
|---|---|---|
| configuration | baseline | baseline-fastembed |
| chunking metadata | off | off |
| chunking max_tokens | default | default |
| Pinakes | 0.32.8 (pypi, code ad0db0547b7e) | 0.32.8 (pypi, code ad0db0547b7e) |

Questions: A has 2556, B has 2556; compared on the 2556 they share. Searches that returned exactly the same passages in the same order with the same scores: 0 of 2556. Ignoring the scores, searches that returned the same articles in the same order: 2424 of 2556 over the whole pool, 2507 of 2556 over the first 8 passages.

Paired on the same answerable questions. "B found, A missed" counts questions; the sign test is exact, two-sided, on those discordant pairs.

| measure | A | B | B found, A missed | A found, B missed | sign test p |
|---|---|---|---|---|---|
| all evidence in first 8 passages | 43.4% | 43.3% | 0 | 1 | 1.0000 |
| all evidence in first 20 passages | 61.0% | 60.9% | 0 | 2 | 0.5000 |
| all evidence in first 5 docs | 43.5% | 43.4% | 0 | 1 | 1.0000 |
| all evidence in first 10 docs | 57.5% | 57.4% | 1 | 4 | 0.3750 |
| any evidence in first 5 passages | 91.5% | 91.5% | 0 | 0 | n/a |

By question type, for *every gold article in the first 8 passages*:

| type | n | A | B | B found, A missed | A found, B missed | sign test p |
|---|---|---|---|---|---|---|
| comparison_query | 856 | 60.3% | 60.3% | 0 | 0 | n/a |
| inference_query | 816 | 25.2% | 25.2% | 0 | 0 | n/a |
| temporal_query | 583 | 43.9% | 43.7% | 0 | 1 | 1.0000 |

Top-1 score AUROC, answerable vs null: A 0.944, B 0.944.

## Do the confidence thresholds transfer between the two runs?

Pinakes applies a stored threshold whenever its `fitted_for` (model and revision) equals the reranker in use ([`confidence_of`](https://github.com/lucagattoni/pinakes/blob/main/src/pinakes/search.py)); it records neither the provider nor the scale of the scores. This labels each run's held-out questions with its own thresholds and with the other run's, as Pinakes would if the two had matched:

| scores of | thresholds of | low below | high above | null labelled high | null labelled low | answerable labelled high | answerable labelled low |
|---|---|---|---|---|---|---|---|
| A | A | 0.0124 | 0.1565 | 16.2% (25 of 154) | 45.5% (70 of 154) | 90.4% (1030 of 1140) | 1.3% (15 of 1140) |
| B | B | -4.3787 | -1.6845 | 16.2% (25 of 154) | 45.5% (70 of 154) | 90.4% (1030 of 1140) | 1.3% (15 of 1140) |
| B | A | 0.0124 | 0.1565 | 9.1% (14 of 154) | 90.3% (139 of 154) | 71.6% (816 of 1140) | 26.7% (304 of 1140) |
| A | B | -4.3787 | -1.6845 | 100.0% (154 of 154) | 0.0% (0 of 154) | 100.0% (1140 of 1140) | 0.0% (0 of 1140) |
