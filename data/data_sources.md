# Data Sources — H17 Fair AI Hiring

## Primary
| # | Source | URL | Use | License |
|---|--------|-----|-----|---------|
| 1 | Synthetic résumé generator | `src/fair_hiring/data.py` | 5,000 candidates with proxy-bias channel | MIT |

## Reference
| Source | URL | Use |
|--------|-----|-----|
| Fairlearn — User Guide | https://fairlearn.org/main/user_guide/ | Equalized-odds postprocessing API |
| Hardt, Price & Srebro (2016) | https://arxiv.org/abs/1610.02413 | Equality of opportunity definition |
| Agarwal et al. (2018) | https://arxiv.org/abs/1803.02453 | Reductions approach to fair classification |
| Barocas, Hardt & Narayanan, *Fairness and Machine Learning* | https://fairmlbook.org/ | Field-defining textbook |
| EEOC Uniform Guidelines (4/5 rule) | https://www.eeoc.gov/laws/regulations/uniform-employee-selection-guidelines-1978 | Selection-rate disparity benchmark |
| UAE Federal Decree-Law No. 33 of 2021 (Labor) | https://u.ae/ | Local labor law context |

## How to regenerate
```bash
python -m fair_hiring.data
python -m fair_hiring.models
```

## Attribution
If publishing audit numbers based on this pipeline, please cite Hardt-Price-Srebro (2016) and the Fairlearn library.
