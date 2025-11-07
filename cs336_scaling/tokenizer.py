from typing import Iterable, Iterator
from functools import partial
import itertools
import logging
import numpy as np
import os
import pickle
import regex as re

from concurrent.futures import ProcessPoolExecutor

from .train_bpe import PAT


class Tokenizer():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str]=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.reverse_vocab = dict(zip(list(self.vocab.values()), list(self.vocab.keys())))
        

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str]=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self.encode_iterable([text])
            
        text_chunks = re.split(
            "(" + "|".join(sorted(map(re.escape, self.special_tokens), reverse=True)) + ")", text) # [doc, special_tokens, doc, ...]
        
        return self.encode_iterable(text_chunks)
    
    
    def encode_special_tokens(self, interleaved_special_tokens: list[str]) -> list[int]:
        return list(map(lambda x: [self.reverse_vocab[x.encode("utf-8")]], interleaved_special_tokens))
    

    def merge_bytes(self, _bytes_batch: list[tuple[bytes]]) -> list[tuple[bytes]]:
        _bytes_batch = _bytes_batch[:]  # copy to avoid mutating the input
        output_bytes_batch = []
        for _bytes in _bytes_batch:
            _bytes = list(_bytes)
            for merge in self.merges:
                i, end = 0, len(_bytes) - 1
                while i < end:
                    if merge == (_bytes[i], _bytes[i + 1]):
                        _bytes[i] = _bytes[i] + _bytes[i + 1]
                        _bytes.pop(i + 1)
                        end -= 1
                    i += 1
            output_bytes_batch.append(tuple(_bytes))
        return output_bytes_batch
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        
        def pre_tokenize(text_chunk: str) -> list[str]:
            if self.special_tokens and text_chunk in self.special_tokens:
                return [text_chunk]
            return re.findall(PAT, text_chunk)
        
        def word_to_bytes(word: str) -> tuple[bytes]:
            if self.special_tokens and word in self.special_tokens:
                return (word.encode("utf-8"),)
            return tuple([bytes([b]) for b in word.encode("utf-8")])

        if self.special_tokens:
            normalized_iterable = []
            for text in iterable:
                # Separate each doc by special tokens.
                normalized_iterable += re.split("(" + "|".join(sorted(map(re.escape, self.special_tokens), reverse=True)) + ")", text)
        else:
            normalized_iterable = iterable
        
        logging.info("Pre-tokenize Starts")
        words = []
        for text_chunk in normalized_iterable:
            words += pre_tokenize(text_chunk) # [w00,w01,...,w10,w11, ...]

        logging.info("Find Unique Words")
        unique_words = list(set(words))

        logging.info("Find Unique Word Bytes")
        unique_word_bytes = list(map(word_to_bytes, unique_words)) # [b00,b01,...,b10,b11, ...]

        logging.info("Build Merge Words Dictionary")
        BATCH_SIZE = 256  # process multiple words in each process to reduce IPC overhead
        batches = [unique_word_bytes[i:i+BATCH_SIZE] for i in range(0, len(unique_word_bytes), BATCH_SIZE)]
        with ProcessPoolExecutor() as executor:
            merged_unique_word_bytes = itertools.chain.from_iterable(executor.map(self.merge_bytes, batches))
        word_to_merged_bytes = dict(zip(unique_words, merged_unique_word_bytes))

        logging.info("Lookup Each Words in the Dictionary")
        _bytes = itertools.chain.from_iterable([word_to_merged_bytes[w] for w in words])

        logging.info("Bytes to IDs")
        return [self.reverse_vocab[b] for b in _bytes] # [i00,i01,...,i10,i11,...], i00 is an integer
    

    def decode(self, ids: list[int]) -> str:
        bytes_ = b''
        for id in ids:
            bytes_ += self.vocab[id]
        return bytes_.decode("utf-8", errors='replace')
    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"  # optional format
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_subdir = "../ckpts/SlimPajama-627B-tokenizer-valid-chunk1-vocab-32k.pkl"
    merge_subdir = "../ckpts/SlimPajama-627B-tokenizer-valid-chunk1-merges-32k.pkl"
    input_subdir = "../data/SlimPajama-627B-valid-chunk1.txt"
    output_subdir = "../data/SlimPajama-627B-valid-chunk1-ids.bin"
    vocab_file_path = os.path.join(current_dir, vocab_subdir)
    merges_file_path = os.path.join(current_dir, merge_subdir)
    input_path = os.path.join(current_dir, input_subdir)
    output_path = os.path.join(current_dir, output_subdir)

    tokenizer = Tokenizer.from_files(vocab_file_path, merges_file_path, ["<|endoftext|>"])
    with open(input_path, "r") as f:
        ids = tokenizer.encode(f.read())
    np.array(ids, dtype=np.int32).tofile(output_path)