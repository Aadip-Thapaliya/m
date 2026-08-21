# Leakage-safe preprocessing protocol

1. Sort by timestamp, preserve original file rows, and document missing timestamps.
2. Split chronologically; never shuffle observations before train/validation/test separation.
3. Estimate scaling parameters on the training prefix alone. For an online scaler, update only after each observation is revealed.
4. Apply any interpolation causally. Future-aware centered interpolation is not valid for streaming evaluation.
5. Define context length, horizon, lag maximum, and forecast issue time explicitly.
6. At a regime boundary, align the graph with the transition that generated the target, not the previous observation index.
7. For ETT, exclude the `date` column from numeric graph estimation and preserve all seven measured variables.
8. Do not overwrite the bundled CC BY-ND datasets. Write transformed research data to ignored local output directories.
9. Store seed, split boundaries, scaler settings, missing-data decisions, graph threshold, and software versions with each run.
10. For `H > 1`, update the model only when each corresponding future label has actually arrived.
