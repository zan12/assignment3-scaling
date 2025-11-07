import json
import os


all_dirs = []

def find_all_files(file_dir):
    for subdir in os.listdir(file_dir):
        subdir = os.path.join(file_dir, subdir)
        if os.path.isdir(subdir):
            find_all_files(subdir)
        elif subdir.endswith(".jsonl"):
            all_dirs.append(subdir)

def write_to_txt(target_dir):
    with open(target_dir, "w") as fo:
        for dir in all_dirs:
            with open(dir, "r") as fi:
                for line in fi:
                    text = json.loads(line)["text"]
                    fo.write(text)
                    fo.write("\n<|endoftext|>\n")


if __name__ == "__main__":
    find_all_files("./SlimPajama-627B/validation/chunk1")
    write_to_txt("./SlimPajama-627B-valid-chunk1.txt")