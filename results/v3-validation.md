# V3 validation results

Each row reports AUROC on the pooled held-out predictions from one stratified five-fold run. V1 and V3 use the same folds for each seed. These repetitions reuse the same 6,941 labeled accounts.

| Split seed | V1 AUROC | V3 AUROC |
| ---: | ---: | ---: |
| 319 | 0.970579 | 0.974753 |
| 441 | 0.970700 | 0.975017 |
| 608 | 0.976118 | 0.980875 |
| 731 | 0.975999 | 0.980585 |
| 896 | 0.978644 | 0.981902 |
| 992 | 0.979061 | 0.982468 |
| 1103 | 0.977804 | 0.981626 |
| 1229 | 0.978067 | 0.980903 |
| 1361 | 0.977314 | 0.981731 |
| 1483 | 0.973742 | 0.979926 |
| 1601 | 0.977350 | 0.980701 |
| 1741 | 0.974923 | 0.980039 |

V3 averaged **0.98004** across all 12 repetitions. On the six additional seeds starting at 1103, it averaged **0.98082**, compared with **0.97653** for V1.

The selected V3 entry scored **0.97676** on the Kaggle public leaderboard. Its CSV contains 2,975 predictions in sample-submission order. The SHA-256 of the reference CSV is:

```text
cc9ec6aa39da233c604564dafa70f697dbe81f6db03a58d23440e16f05e78a77
```

The checksum refers to the reference run with Python 3.10 on macOS. Numerical libraries on another platform may produce small differences in the last decimal places.
