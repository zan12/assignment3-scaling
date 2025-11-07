import time
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    # TODO: auto-launch when incurring rate limit errors
    while True:
        try:
            snapshot_download(
                repo_id="cerebras/SlimPajama-627B",
                repo_type="dataset",
                allow_patterns="validation/*", # f"train/chunk1/*",
                local_dir="./SlimPajama-627B",
                local_dir_use_symlinks=False,
                max_workers=8,
            )
        except:
            time.sleep(300)
