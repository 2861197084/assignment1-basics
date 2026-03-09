import os
import regex as re
from typing import BinaryIO
from multiprocessing import process, get_context
from collections import defaultdict

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_COMPILED_PAT = re.compile(GPT2_PAT)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    process_num: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 1. Initialization 
    vocab_init = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab_init[len(vocab_init)] = tok.encode("utf-8")

    # 2. Pre-tokenization
    with open(input_path, "rb") as f:
        bounds = find_chunk_boundaries(f, process_num, "<|endoftext|>".encode("utf-8"))
    chunks_init = [
        (input_path, start, end, special_tokens)
        for start, end in zip(bounds[:-1], bounds[1:])
    ]
    with get_context("spawn").Pool(processes=process_num) as pool:
        chunk_results = pool.map(process_chunk, chunks_init)

    # 3. Compute BPE merges
    ids: list[list[int]] = [
        token_ids for chunk_ids in chunk_results for token_ids in chunk_ids
    ]
    merges: list[tuple[int, int]] = []
    pair_to_indices, counts = _get_pair_counts(ids)

    merge_num = vocab_size - len(vocab_init)
    for i in range(merge_num):
        if not counts:
            break
        
        # Find the most frequent pair
        def rank(pair: tuple[int, int]) -> tuple[int, tuple[bytes, bytes]]:
            return counts[pair],(vocab_init[pair[0]], vocab_init[pair[1]])
        max_pair = max(counts, key=rank)
        new_token = vocab_init[max_pair[0]] + vocab_init[max_pair[1]]
        new_id = len(vocab_init)
        vocab_init[new_id] = new_token
        merges.append(max_pair)

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token_ids = ids[j]
            if len(token_ids) < 2:
                continue

            for pair in zip(token_ids, token_ids[1:]):
                counts[pair] -= 1
                pair_to_indices[pair].discard(j)
                if counts[pair] == 0:
                    del counts[pair]
                    del pair_to_indices[pair]
                    
            new_token_ids = _merge_pair(token_ids, max_pair, new_id)

            for pair in zip(new_token_ids, new_token_ids[1:]):
                counts[pair] += 1
                pair_to_indices[pair].add(j)

            ids[j] = new_token_ids

    merges = [(vocab_init[a], vocab_init[b]) for a, b in merges]
    return vocab_init, merges


def _get_pair_counts(
    ids: list[list[int]]
) -> tuple[
        defaultdict[tuple[int, int], set],
        defaultdict[tuple[int, int], int]
    ]:
    pair_to_indices = defaultdict(set)
    counts = defaultdict(int)
    for i, token_ids in enumerate(ids):
        for pair in zip(token_ids, token_ids[1:]):
            pair_to_indices[pair].add(i)
            counts[pair] += 1
    return pair_to_indices, counts


def _merge_pair(
    token_ids: list[int], pair: tuple[int, int], new_id: int
) -> list[int]:
    new_token_ids = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == pair:
            new_token_ids.append(new_id)
            i += 2
        else:
            new_token_ids.append(token_ids[i]) 
            i += 1
    return new_token_ids


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes
) -> list[int]:
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )
    file.seek(0, os.SEEK_END)    
    file_size = file.tell()
    file.seek(0)

    chunk_size = max(1, (file_size // desired_num_chunks))
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        pos = chunk_boundaries[bi]
        file.seek(pos)
        while True:
            mini_chunk = file.read(mini_chunk_size)

            if not mini_chunk:
                chunk_boundaries[bi] = file_size
                break

            found_bound = mini_chunk.find(split_special_token)
            if found_bound != -1:
                chunk_boundaries[bi] = pos + found_bound
                break
            pos += mini_chunk_size
    
    return sorted(set(chunk_boundaries))


def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[int]]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode(encoding="utf-8", errors="ignore")
    # Normalize line endings so tokenization is stable across OS checkouts.
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    documents = re.split(pattern, chunk)
    
    chunk_ids : list[list[int]] = []
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in _COMPILED_PAT.finditer(doc)]
        chunk_ids.extend([list(token) for token in tokens])

    return chunk_ids


