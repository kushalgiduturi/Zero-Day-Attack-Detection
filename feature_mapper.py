import pandas as pd
import numpy as np
from universal_features import UNIVERSAL_FEATURES

# Aliases — same feature, different names across datasets
FEATURE_ALIASES = {
    "FLOW_DURATION_MILLISECONDS": [
        "duration", "flow_duration", "Duration",
        "FLOW_DURATION", "fl_dur"
    ],
    "TOTAL_FWDPACKETS": [
        "total_fwd_packets", "fwd_packets", "Tot Fwd Pkts",
        "src_bytes", "fwd_pkts_tot"
    ],
    "TOTAL_BWDPACKETS": [
        "total_bwd_packets", "bwd_packets", "Tot Bwd Pkts",
        "dst_bytes", "bwd_pkts_tot"
    ],
    "FLOW_BYTES_PER_SECOND": [
        "flow_byts_s", "bytes_per_sec", "Flow Bytes/s",
        "byterate"
    ],
    "FLOW_PACKETS_PER_SECOND": [
        "flow_pkts_s", "packets_per_sec", "Flow Pkts/s",
        "pktrate"
    ],
    "FWDPACKET_LENGTH_MEAN": [
        "fwd_pkt_len_mean", "Fwd Pkt Len Mean",
        "fwd_seg_size_avg"
    ],
    "FLOW_IAT_MEAN": [
        "flow_iat_mean", "Flow IAT Mean",
        "iat_mean", "mean_iat"
    ],
    # ... add more aliases as needed
}


class UniversalFeatureMapper:
    """
    Automatically maps any dataset's columns
    to the universal 20 core features.
    Missing features are filled with 0.
    """

    def __init__(self):
        self.mapping      = {}   # new_col → universal_col
        self.missing_cols = []   # features not found in dataset

    def fit(self, df: pd.DataFrame) -> "UniversalFeatureMapper":
        """
        Auto-detect which columns in df match universal features.
        """
        df_cols_lower = {c.lower(): c for c in df.columns}
        self.mapping  = {}

        for universal_col, aliases in FEATURE_ALIASES.items():
            # Check exact match first
            if universal_col in df.columns:
                self.mapping[universal_col] = universal_col
                continue

            # Check aliases
            found = False
            for alias in aliases:
                if alias.lower() in df_cols_lower:
                    self.mapping[df_cols_lower[alias.lower()]] = universal_col
                    found = True
                    break

            if not found:
                self.missing_cols.append(universal_col)

        print(f"  Mapped    : {len(self.mapping)} features")
        print(f"  Missing   : {len(self.missing_cols)} features "
              f"(will be filled with 0)")
        if self.missing_cols:
            print(f"  Missing cols: {self.missing_cols}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with exactly the universal features.
        """
        # Rename matched columns
        result = df.rename(columns=self.mapping)

        # Add missing columns as 0
        for col in self.missing_cols:
            result[col] = 0.0

        # Return only universal features in consistent order
        return result[UNIVERSAL_FEATURES].astype(np.float32)

    def transform_dict(self, flow: dict) -> dict:
        """Transform a single flow dict."""
        mapped = {}
        for original, universal in self.mapping.items():
            mapped[universal] = flow.get(original, 0.0)
        for col in self.missing_cols:
            mapped[col] = 0.0
        return {f: mapped.get(f, 0.0) for f in UNIVERSAL_FEATURES}