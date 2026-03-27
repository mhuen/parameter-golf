#!/usr/bin/env python3
"""
Build a KenLM n-gram language model from tokenized training data.

Supports two modes:
  - Token mode (default): SentencePiece-tokenized sp1024 data, sentences
    split on BOS token (id=1).
  - Byte mode (--byte-mode): Raw UTF-8 byte data (vocab 256), sentences
    created by chunking the byte stream into fixed-length segments.

Steps:
  1. Load binary token shards (and SentencePiece tokenizer in token mode).
  2. Write a text corpus (one "sentence" per line, space-separated IDs).
  3. Run KenLM's `lmplz` to estimate an ARPA n-gram model.
  4. Run KenLM's `build_binary` to compile a fast binary model.
  5. Optionally evaluate on validation data and report perplexity.

Usage:
  python build_kenlm.py [--order 5] [--max-tokens 200000000] [--eval]
  python build_kenlm.py --byte-mode [--order 6] [--sentence-len 512] [--eval]
  python build_kenlm.py --efficient-byte-mode [--order 6] [--sentence-len 512] [--eval]
"""

import argparse
import glob
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DATA_PATH_SP1024 = "./data/datasets/fineweb10B_sp1024"
DATA_PATH_BYTES = "./data/datasets/fineweb10B_bytes"
TOKENIZER_PATH = "./data/tokenizers/fineweb_1024_bpe.model"
KENLM_BIN = (
    "/private/var/lib/persistent-storage/mirco/test_bed/language_models/kenlm/build/bin"
)
OUTPUT_DIR_SP1024 = "./models/kenlm"
OUTPUT_DIR_BYTES = "./models/kenlm_bytes"
OUTPUT_DIR_EFFICIENT_BYTES = "./models/kenlm_efficient_bytes"

BOS_ID = 1  # SentencePiece BOS token id
BYTE_SENTENCE_LEN = 512  # default chunk size for byte-mode sentences


# ---------------------------------------------------------------------------
# Data loading (mirrors train_gpt.py)
# ---------------------------------------------------------------------------
def load_data_shard(file: Path) -> np.ndarray:
    """Load a single binary shard and return token IDs as a numpy uint16 array."""
    header_bytes = 256 * 4  # 256 int32 header entries
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return tokens


def load_all_tokens(pattern: str, max_tokens: int = 0) -> np.ndarray:
    """Load token IDs from all matching shards."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    chunks = []
    total = 0
    for f in files:
        tokens = load_data_shard(Path(f))
        if max_tokens > 0 and total + tokens.size > max_tokens:
            tokens = tokens[: max_tokens - total]
        chunks.append(tokens)
        total += tokens.size
        if max_tokens > 0 and total >= max_tokens:
            break
    return np.concatenate(chunks)


# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------
def write_text_corpus(
    token_ids: np.ndarray,
    output_path: str,
) -> int:
    """
    Write token IDs as whitespace-separated strings, one sentence per line.
    Sentences are split on the BOS token (id=1).  The BOS token itself is
    omitted — KenLM inserts its own <s>/</s> sentence markers implicitly.
    Returns the number of sentences written.
    """
    n_sentences = 0

    with open(output_path, "w", encoding="utf-8") as f:
        current_ids: list[str] = []
        for tid in token_ids:
            if tid == BOS_ID:
                if current_ids:
                    f.write(" ".join(current_ids) + "\n")
                    n_sentences += 1
                current_ids = []
            else:
                current_ids.append(str(tid))
        # flush last sentence
        if current_ids:
            f.write(" ".join(current_ids) + "\n")
            n_sentences += 1

    return n_sentences


def write_text_corpus_bytes(
    token_ids: np.ndarray,
    output_path: str,
    sentence_len: int = BYTE_SENTENCE_LEN,
) -> int:
    """
    Write byte-valued token IDs as whitespace-separated strings, one sentence
    per line.  Since byte data has no document boundaries, we chunk into
    fixed-length segments of `sentence_len` bytes each.
    Returns the number of sentences written.
    """
    n_sentences = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for start in range(0, len(token_ids), sentence_len):
            chunk = token_ids[start : start + sentence_len]
            f.write(" ".join(str(int(b)) for b in chunk) + "\n")
            n_sentences += 1
    return n_sentences


def _split_into_sentences_bytes(
    token_ids: np.ndarray, sentence_len: int = BYTE_SENTENCE_LEN
) -> list[list[str]]:
    """Split byte token IDs into fixed-length sentences for evaluation."""
    sentences: list[list[str]] = []
    for start in range(0, len(token_ids), sentence_len):
        chunk = token_ids[start : start + sentence_len]
        sentences.append([str(int(b)) for b in chunk])
    return sentences


# ---------------------------------------------------------------------------
# KenLM training
# ---------------------------------------------------------------------------
def run_lmplz(
    corpus_path: str,
    arpa_path: str,
    order: int,
    kenlm_bin: str,
    memory: str = "50%",
) -> None:
    """Run KenLM's lmplz to estimate an ARPA model."""
    lmplz = os.path.join(kenlm_bin, "lmplz")
    cmd = [
        lmplz,
        "-o",
        str(order),
        "--memory",
        memory,
        "--text",
        corpus_path,
        "--arpa",
        arpa_path,
        # Treat each line as a sentence (no BOS/EOS insertion by lmplz,
        # since our corpus already uses SentencePiece BOS tokens).
        "--discount_fallback",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_build_binary(
    arpa_path: str,
    binary_path: str,
    kenlm_bin: str,
) -> None:
    """Compile ARPA model to KenLM binary format."""
    build_binary = os.path.join(kenlm_bin, "build_binary")
    cmd = [build_binary, arpa_path, binary_path]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _split_into_sentences(token_ids: np.ndarray) -> list[list[str]]:
    """Split token IDs into sentences on BOS, returning lists of ID strings.
    BOS tokens are omitted (KenLM adds its own <s>/</s> markers)."""
    sentences: list[list[str]] = []
    current: list[str] = []
    for tid in token_ids:
        if tid == BOS_ID:
            if current:
                sentences.append(current)
            current = []
        else:
            current.append(str(tid))
    if current:
        sentences.append(current)
    return sentences


def evaluate_kenlm(
    binary_path: str,
    val_token_ids: np.ndarray,
    kenlm_bin: str,
) -> None:
    """Evaluate KenLM model on validation data and print perplexity."""
    try:
        import kenlm as kenlm_py

        model = kenlm_py.Model(binary_path)
    except ImportError:
        print(
            "kenlm Python bindings not installed. "
            "Falling back to `query` binary for evaluation."
        )
        _evaluate_kenlm_query(binary_path, val_token_ids, kenlm_bin)
        return

    sentences = _split_into_sentences(val_token_ids)

    total_log_prob = 0.0
    total_tokens = 0
    for sent_ids in sentences:
        text = " ".join(sent_ids)
        # model.score returns log10 probability
        log10_prob = model.score(text, bos=True, eos=True)
        total_log_prob += log10_prob
        total_tokens += len(sent_ids)

    # Perplexity = 10^(-avg_log10_prob)
    avg_log10 = total_log_prob / total_tokens
    ppl = 10 ** (-avg_log10)
    # Cross-entropy in nats
    ce_nats = -avg_log10 * math.log(10)
    ce_bits = -avg_log10 * math.log2(10)

    print(f"\n=== KenLM Evaluation ===")
    print(f"  Sentences:      {len(sentences):,}")
    print(f"  Tokens:         {total_tokens:,}")
    print(f"  Perplexity:     {ppl:.2f}")
    print(f"  Cross-entropy:  {ce_nats:.4f} nats  ({ce_bits:.4f} bits)")


def evaluate_kenlm_bytes(
    binary_path: str,
    val_token_ids: np.ndarray,
    kenlm_bin: str,
    sentence_len: int = BYTE_SENTENCE_LEN,
) -> None:
    """Evaluate KenLM model on byte-level validation data."""
    try:
        import kenlm as kenlm_py

        model = kenlm_py.Model(binary_path)
    except ImportError:
        print(
            "kenlm Python bindings not installed. "
            "Falling back to `query` binary for evaluation."
        )
        sentences = _split_into_sentences_bytes(val_token_ids, sentence_len)
        _evaluate_kenlm_query_sentences(binary_path, sentences, kenlm_bin)
        return

    sentences = _split_into_sentences_bytes(val_token_ids, sentence_len)

    total_log_prob = 0.0
    total_tokens = 0
    for sent_ids in sentences:
        text = " ".join(sent_ids)
        log10_prob = model.score(text, bos=True, eos=True)
        total_log_prob += log10_prob
        total_tokens += len(sent_ids)

    avg_log10 = total_log_prob / total_tokens
    ppl = 10 ** (-avg_log10)
    ce_nats = -avg_log10 * math.log(10)
    ce_bits = -avg_log10 * math.log2(10)

    print(f"\n=== KenLM Byte-Mode Evaluation ===")
    print(f"  Sentences:      {len(sentences):,}")
    print(f"  Bytes:          {total_tokens:,}")
    print(f"  Perplexity:     {ppl:.2f}")
    print(f"  Cross-entropy:  {ce_nats:.4f} nats  ({ce_bits:.4f} bits/byte)")


def _evaluate_kenlm_query(
    binary_path: str,
    val_token_ids: np.ndarray,
    kenlm_bin: str,
) -> None:
    """Fallback evaluation using the `query` binary."""
    sentences = _split_into_sentences(val_token_ids)
    _evaluate_kenlm_query_sentences(binary_path, sentences, kenlm_bin)


def _evaluate_kenlm_query_sentences(
    binary_path: str,
    sentences: list[list[str]],
    kenlm_bin: str,
) -> None:
    """Fallback evaluation using the `query` binary with pre-split sentences."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as tmp:
        for sent_ids in sentences:
            tmp.write(" ".join(sent_ids) + "\n")
        tmp_path = tmp.name

    query_bin = os.path.join(kenlm_bin, "query")
    cmd = [query_bin, binary_path]
    print(f"Running: {' '.join(cmd)} < {tmp_path}")
    result = subprocess.run(cmd, stdin=open(tmp_path), capture_output=True, text=True)
    os.unlink(tmp_path)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.stderr:
        print(result.stderr[-500:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Build a KenLM n-gram model")
    parser.add_argument(
        "--order", type=int, default=5, help="N-gram order (default: 5)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200_000_000,
        help="Max training tokens to use (default: 200M, 0=all)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to data shards (auto-selected based on mode if omitted)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=TOKENIZER_PATH, help="SentencePiece model"
    )
    parser.add_argument(
        "--kenlm-bin", type=str, default=KENLM_BIN, help="Path to KenLM binaries"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (auto-selected based on mode if omitted)",
    )
    parser.add_argument(
        "--memory", type=str, default="50%", help="Memory for lmplz (e.g. '4G', '50%%')"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate on validation data after building"
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="Only generate the text corpus, don't train",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--byte-mode",
        action="store_true",
        help="Use raw byte data (vocab 256) instead of SentencePiece tokens",
    )
    mode_group.add_argument(
        "--efficient-byte-mode",
        action="store_true",
        help="Use efficient byte tokenizer (vocab 208) — remaps raw byte shards",
    )
    parser.add_argument(
        "--sentence-len",
        type=int,
        default=BYTE_SENTENCE_LEN,
        help=f"Sentence length for byte mode chunking (default: {BYTE_SENTENCE_LEN})",
    )
    parser.add_argument(
        "--keep",
        type=str,
        nargs="+",
        default=None,
        help="ByteCategory(s) to keep (e.g. letter digit). Only with --efficient-byte-mode.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        nargs="+",
        default=None,
        help="ByteCategory(s) to fold (e.g. uppercase). Only with --efficient-byte-mode.",
    )
    parser.add_argument(
        "--other-strategy",
        type=str,
        default="drop",
        choices=["drop", "pad", "boundary"],
        help="How to handle non-kept tokens (default: drop). Only with --efficient-byte-mode.",
    )
    args = parser.parse_args()

    # Resolve defaults based on mode
    any_byte_mode = args.byte_mode or args.efficient_byte_mode
    if args.data_path is None:
        args.data_path = DATA_PATH_BYTES if any_byte_mode else DATA_PATH_SP1024
    if args.output_dir is None:
        if args.efficient_byte_mode:
            suffix = ""
            if args.keep:
                suffix += "_" + "_".join(sorted(args.keep))
            if args.fold:
                suffix += "_fold_" + "_".join(sorted(args.fold))
            args.output_dir = OUTPUT_DIR_EFFICIENT_BYTES + suffix
        elif args.byte_mode:
            args.output_dir = OUTPUT_DIR_BYTES
        else:
            args.output_dir = OUTPUT_DIR_SP1024

    os.makedirs(args.output_dir, exist_ok=True)

    # Paths
    corpus_path = os.path.join(args.output_dir, "corpus.txt")
    arpa_path = os.path.join(args.output_dir, f"kenlm_{args.order}gram.arpa")
    binary_path = os.path.join(args.output_dir, f"kenlm_{args.order}gram.binary")

    if args.efficient_byte_mode:
        from efficient_byte_tokenizer import EfficientByteTokenizer, ByteCategory, OtherTokenStrategy
        tok_kwargs: dict = {}
        if args.keep:
            tok_kwargs["keep"] = frozenset(ByteCategory(k) for k in args.keep)
        if args.fold:
            tok_kwargs["fold"] = frozenset(ByteCategory(f) for f in args.fold)
        if args.keep:
            tok_kwargs["other"] = OtherTokenStrategy(args.other_strategy)
        eff_tok = EfficientByteTokenizer(**tok_kwargs)
        print(f"  {eff_tok.describe()}")
        print(f"  sentence_len={args.sentence_len}")
    elif args.byte_mode:
        print(f"Byte mode: vocab=256, sentence_len={args.sentence_len}")
    else:
        import sentencepiece as spm
        print(f"Loading tokenizer: {args.tokenizer}")
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
        print(f"  Vocab size: {sp.vocab_size()}")

    # Step 1: Generate text corpus (skip if it already exists)
    if os.path.exists(corpus_path):
        corpus_size = os.path.getsize(corpus_path)
        print(
            f"\nCorpus already exists: {corpus_path} ({corpus_size / 1e9:.2f} GB), skipping."
        )
    else:
        train_pattern = f"{args.data_path}/fineweb_train_*.bin"
        print(f"\nLoading training tokens from: {train_pattern}")
        print(
            f"  Max tokens: {args.max_tokens:,}"
            if args.max_tokens
            else "  Using all tokens"
        )
        train_tokens = load_all_tokens(train_pattern, max_tokens=args.max_tokens)
        print(f"  Loaded {train_tokens.size:,} tokens")
        if args.efficient_byte_mode:
            train_tokens = eff_tok.filter_stream(
                eff_tok.remap_byte_array(train_tokens.astype(np.uint8))
            )
            print(f"  Remapped + filtered: {train_tokens.size:,} tokens (vocab {eff_tok.vocab_size})")

        print(f"\nWriting text corpus to: {corpus_path}")
        if any_byte_mode:
            n_sentences = write_text_corpus_bytes(
                train_tokens, corpus_path, args.sentence_len
            )
        else:
            n_sentences = write_text_corpus(train_tokens, corpus_path)
        corpus_size = os.path.getsize(corpus_path)
        print(f"  {n_sentences:,} sentences, {corpus_size / 1e9:.2f} GB")

    if args.corpus_only:
        print("Done (corpus only).")
        return

    # Step 2: Train KenLM model
    print(f"\n{'=' * 60}")
    print(f"Training {args.order}-gram KenLM model...")
    print(f"{'=' * 60}")
    run_lmplz(corpus_path, arpa_path, args.order, args.kenlm_bin, args.memory)
    arpa_size = os.path.getsize(arpa_path)
    print(f"  ARPA model: {arpa_size / 1e6:.1f} MB")

    # Step 3: Build binary
    print(f"\nBuilding binary model...")
    run_build_binary(arpa_path, binary_path, args.kenlm_bin)
    binary_size = os.path.getsize(binary_path)
    print(f"  Binary model: {binary_size / 1e6:.1f} MB")

    # Step 4: Evaluate
    if args.eval:
        print(f"\n{'=' * 60}")
        print("Evaluating on validation data...")
        print(f"{'=' * 60}")
        val_pattern = f"{args.data_path}/fineweb_val_*.bin"
        val_files = sorted(glob.glob(val_pattern))
        if not val_files and any_byte_mode:
            # No dedicated val split — use the last training shard
            train_files = sorted(glob.glob(f"{args.data_path}/fineweb_train_*.bin"))
            if not train_files:
                raise FileNotFoundError(f"No shards found in {args.data_path}")
            val_file = train_files[-1]
            print(f"  No val split found; using last train shard as holdout: {val_file}")
            val_tokens = load_data_shard(Path(val_file))
        else:
            val_tokens = load_all_tokens(val_pattern)
        if args.efficient_byte_mode:
            val_tokens = eff_tok.filter_stream(
                eff_tok.remap_byte_array(val_tokens.astype(np.uint8))
            )
        print(f"  Loaded {val_tokens.size:,} validation tokens")
        if any_byte_mode:
            evaluate_kenlm_bytes(
                binary_path, val_tokens, args.kenlm_bin, args.sentence_len
            )
        else:
            evaluate_kenlm(binary_path, val_tokens, args.kenlm_bin)

    print(f"\nDone! Model files:")
    print(f"  ARPA:   {arpa_path}")
    print(f"  Binary: {binary_path}")


if __name__ == "__main__":
    main()
