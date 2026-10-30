# `20261030_1826-baseline-pypi-0.32.8` (A) against `20261030_1910-metadata-prefix-pypi-0.32.8` (B)

|  | A | B |
|---|---|---|
| configuration | baseline | metadata-prefix |
| chunking metadata | off | prefix |
| chunking max_tokens | default | 459 |
| Pinakes | 0.32.8 (pypi, code ad0db0547b7e) | 0.32.8 (pypi, code ad0db0547b7e) |

Questions: A has 2556, B has 2556; compared on the 2556 they share. Searches that returned exactly the same passages in the same order with the same scores: 5 of 2556.

Paired on the same answerable questions. "B found, A missed" counts questions; the sign test is exact, two-sided, on those discordant pairs.

| measure | A | B | B found, A missed | A found, B missed | sign test p |
|---|---|---|---|---|---|
| all evidence in first 8 passages | 43.4% | 42.3% | 79 | 104 | 0.0758 |
| all evidence in first 20 passages | 61.0% | 55.7% | 71 | 191 | <0.0001 |
| all evidence in first 5 docs | 43.5% | 43.8% | 108 | 101 | 0.6782 |
| all evidence in first 10 docs | 57.5% | 52.8% | 72 | 179 | <0.0001 |
| any evidence in first 5 passages | 91.5% | 91.4% | 46 | 47 | 1.0000 |

By question type, for *every gold article in the first 8 passages*:

| type | n | A | B | B found, A missed | A found, B missed | sign test p |
|---|---|---|---|---|---|---|
| comparison_query | 856 | 60.3% | 59.0% | 39 | 50 | 0.2891 |
| inference_query | 816 | 25.2% | 25.2% | 24 | 24 | 1.0000 |
| temporal_query | 583 | 43.9% | 41.5% | 16 | 30 | 0.0541 |

Top-1 score AUROC, answerable vs null: A 0.944, B 0.944.

## Read without the articles the two configurations chunk differently

The configurations differ in `max_tokens`, which splits 11 articles differently (found by comparing the two indexes' chunk boundaries). The 48 answerable questions whose gold includes one of them are set aside, leaving 2207; any difference left is not due to those articles.

| measure | A | B | B found, A missed | A found, B missed | sign test p |
|---|---|---|---|---|---|
| all evidence in first 8 passages | 43.4% | 42.3% | 78 | 103 | 0.0741 |
| all evidence in first 20 passages | 61.4% | 55.9% | 69 | 190 | <0.0001 |
| all evidence in first 5 docs | 43.5% | 43.8% | 106 | 100 | 0.7277 |
| all evidence in first 10 docs | 57.9% | 53.0% | 71 | 178 | <0.0001 |
| any evidence in first 5 passages | 91.5% | 91.3% | 42 | 46 | 0.7493 |
