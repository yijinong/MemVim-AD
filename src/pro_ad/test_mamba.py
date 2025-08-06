from transformers import AutoModel

MAMBA_PATH = "/data2/yijin/MambaVision-L3-512-21K"


def main() -> None:
    mamba_model = AutoModel.from_pretrained(
        MAMBA_PATH, trust_remote_code=True, local_files_only=True
    )
    # print(f"Loaded MambaVision model from {MODEL_PATH}")

    print(f"Model config: {mamba_model.config.dim}")

    # print(f" Mean: {mamba_model.config.mean}")
    # print(f" Std: {mamba_model.config.std}")
    # print(f" Crop mode: {mamba_model.config.crop_mode}")
    # print(f" Crop pct: {mamba_model.config.crop_pct}")


if __name__ == "__main__":
    main()
