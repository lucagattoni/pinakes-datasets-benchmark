# `20261030_1826-baseline-pypi-0.32.8` (A) against `20261030_2030-baseline-git-b1bad68` (B)

|  | A | B |
|---|---|---|
| configuration | baseline | baseline |
| chunking metadata | off | off |
| chunking max_tokens | default | default |
| Pinakes | 0.32.8 (pypi, code ad0db0547b7e) | 0.32.8 (git, code ad0db0547b7e) |

Questions: A has 2556, B has 200; compared on the 200 they share. Searches that returned exactly the same passages in the same order with the same scores: 200 of 200.

Paired on the same answerable questions. "B found, A missed" counts questions; the sign test is exact, two-sided, on those discordant pairs.

| measure | A | B | B found, A missed | A found, B missed | sign test p |
|---|---|---|---|---|---|
| all evidence in first 8 passages | 42.4% | 42.4% | 0 | 0 | n/a |
| all evidence in first 20 passages | 59.9% | 59.9% | 0 | 0 | n/a |
| all evidence in first 5 docs | 42.4% | 42.4% | 0 | 0 | n/a |
| all evidence in first 10 docs | 55.9% | 55.9% | 0 | 0 | n/a |
| any evidence in first 5 passages | 88.1% | 88.1% | 0 | 0 | n/a |

By question type, for *every gold article in the first 8 passages*:

| type | n | A | B | B found, A missed | A found, B missed | sign test p |
|---|---|---|---|---|---|---|
| comparison_query | 66 | 65.2% | 65.2% | 0 | 0 | n/a |
| inference_query | 67 | 26.9% | 26.9% | 0 | 0 | n/a |
| temporal_query | 44 | 31.8% | 31.8% | 0 | 0 | n/a |

Top-1 score AUROC, answerable vs null: A 0.970, B 0.970.
