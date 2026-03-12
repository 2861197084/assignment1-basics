import regex as re


def process_chunk(
        chunk: str, 
        special_tokens: list[str], 
        keep_special_tokens: bool
        ) -> list[list[bytes]]:
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    if pattern and keep_special_tokens:
        pattern = f"({pattern})"
    segments = re.split(pattern, chunk) if pattern else [chunk]

    pre_token_bytes: list[list[bytes]] = []
    GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for segment in segments:
        if keep_special_tokens and segment in special_tokens:
            token_bytes = [segment.encode("utf-8")]
            pre_token_bytes.append(token_bytes)
        else:
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(GPT2_PAT, segment)]
            for token in tokens:
                token_bytes = [bytes([b]) for b in token]
                pre_token_bytes.append(token_bytes)
    return pre_token_bytes


def _merge_pair(
        token_bytes: list[list[bytes]],
        pair: tuple[bytes, bytes],
        new_token: bytes
) -> list[bytes]:
    new_token_bytes = []
    i = 0
    while i < len(token_bytes):
        if i < len(token_bytes) - 1 and (token_bytes[i], token_bytes[i+1]) == pair:
          new_token_bytes.append(new_token)
          i += 2
        else:
            new_token_bytes.append(token_bytes[i])
            i += 1
    return new_token_bytes  


class Tokenizer:
    def __init__(self, 
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
        ):
        self.vocab = vocab
        self.vocab_reversed = {v: k for k,v in self.vocab.items()}

        self.merges = merges
        self.merges_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))
    
    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
        ) -> "Tokenizer":
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            for line in f:
                id_str, token_str = line.strip().split("\t")
                vocab[int(id_str)] = eval(token_str).encode("utf-8")

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((eval(parts[0]).encode("utf-8"), eval(parts[1]).encode("utf-8")))
        return cls(vocab=vocab, merges=merges, special_tokens = special_tokens)

    def encode(self, text: str) -> list[int]:
        token_ids = []
        pre_tokens_list = process_chunk(text, self.special_tokens, True)

        for tokens in pre_tokens_list:
            if len(tokens) == 1 and tokens[0] in self.vocab_reversed:
                token_id = self.vocab_reversed.get(tokens[0])
                if token_id is not None:
                    token_ids.append(token_id)
                    continue

            while len(tokens) >= 2:
                pairs = list(zip(tokens, tokens[1:]))
                best_pair = min(pairs, key=lambda p: self.merges_ranks.get(p, float('inf')))

                if best_pair not in self.merges_ranks:
                    break

                new_tok = best_pair[0] + best_pair[1]
                tokens = _merge_pair(tokens, best_pair, new_tok)

            for token in tokens:
                token_id = self.vocab_reversed.get(token)
                if token_id is not None:
                    token_ids.append(token_id)

        return token_ids

    def encode_iterable(self, iterable: list[str]) -> iter:
        for line in iterable:
            token_ids = self.encode(line)
            yield from token_ids

    def decode(self, ids: list[int]) -> str:
        tokens = b"".join(self.vocab.get(token_id, b'\xef\xbf\xbd') for token_id in ids)
        return tokens.decode(encoding="utf-8", errors="replace")