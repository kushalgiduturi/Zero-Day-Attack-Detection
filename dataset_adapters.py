class NSLKDDAdapter:
    """Converts NSL-KDD format to universal features."""
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Label"] = (df["label"] != "normal").astype(int)
        df.rename(columns={
            "duration"  : "FLOW_DURATION_MILLISECONDS",
            "src_bytes" : "TOTAL_LENGTH_OF_FWD_PACKETS",
            "dst_bytes" : "TOTAL_LENGTH_OF_BWD_PACKETS",
        }, inplace=True)
        return df


class UNSWNB15Adapter:
    """Converts UNSW-NB15 format to universal features."""
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Label"] = df["label"]   # already 0/1
        df.rename(columns={
            "dur"    : "FLOW_DURATION_MILLISECONDS",
            "spkts"  : "TOTAL_FWDPACKETS",
            "dpkts"  : "TOTAL_BWDPACKETS",
            "sbytes" : "TOTAL_LENGTH_OF_FWD_PACKETS",
            "dbytes" : "TOTAL_LENGTH_OF_BWD_PACKETS",
            "rate"   : "FLOW_PACKETS_PER_SECOND",
        }, inplace=True)
        return df


class CICIDS2017Adapter:
    """Converts CIC-IDS2017 format to universal features."""
    def convert(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Label"] = (df[" Label"] != "BENIGN").astype(int)
        df.rename(columns={
            " Flow Duration"       : "FLOW_DURATION_MILLISECONDS",
            " Total Fwd Packets"   : "TOTAL_FWDPACKETS",
            " Total Bwd packets"   : "TOTAL_BWDPACKETS",
            " Flow Bytes/s"        : "FLOW_BYTES_PER_SECOND",
            " Flow Packets/s"      : "FLOW_PACKETS_PER_SECOND",
            " Flow IAT Mean"       : "FLOW_IAT_MEAN",
        }, inplace=True)
        return df


# Factory — auto select adapter
ADAPTERS = {
    "nsl-kdd"    : NSLKDDAdapter,
    "unsw-nb15"  : UNSWNB15Adapter,
    "cic-ids2017": CICIDS2017Adapter,
    "nf-uq-nids" : None,   # native format, no adapter needed
}

def get_adapter(dataset_name: str):
    cls = ADAPTERS.get(dataset_name.lower())
    return cls() if cls else None