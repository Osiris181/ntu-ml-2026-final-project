# Global market data

`global_data.csv` is the supplied dated table of `^VIX`, `^IXIC`, `^GSPC`, `^TWII`, `^KS11`, and `TX_OI`. The shared loader reads and validates it; the F0 builder uses the first four series for the agreed global features.

For each feature, the builder independently chooses the latest nonmissing observation dated **strictly before** the prediction date. It treats that feature as missing when the observation is more than five calendar days old. The CSV dates do not verify publication times; see the [central data-quality section](../../README.md#data-quality-and-timing-assumptions). Leave the supplied CSV unchanged.
