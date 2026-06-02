# DCT-Cauvis Result Summary

## Scope
- Benchmark: S-DGOD / Diverse Weather
- Metric: mAP@50 with SDGOD evaluator (batch_size=1, corrected protocol)

## Configured Comparison

| Metric | Cauvis (baseline) | DCT-Cauvis (12-epoch) | Delta |
|---|---:|---:|---:|
| Daytime_Sunny_AP50 | 0.7410 | 0.7530 | +0.0120 |
| Daytime-Foggy_AP50 | 0.5650 | 0.5710 | +0.0060 |
| Dusk-rainy_AP50 | 0.6500 | 0.6420 | -0.0080 |
| Night_rainy_AP50 | 0.4750 | 0.4800 | +0.0050 |
| Night-Sunny_AP50 | 0.6060 | 0.6190 | +0.0130 |
| **target_mean_AP50** | **0.5740** | **0.5780** | **+0.0040** |
| **mean_AP50** | **0.6074** | **0.6130** | **+0.0056** |

## Conclusion
- DCT-Cauvis shows a modest mean AP50 gain under this protocol.
- Stronger claims (e.g., +1.0 target mean AP50 gain and universal target-domain gains) were not met in this verification run.
