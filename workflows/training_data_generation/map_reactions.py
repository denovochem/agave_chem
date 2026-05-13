"""
map_reactions.py — Parallel reaction atom-mapping using TemplateReactionMapper.

Reads reaction SMILES (one per line) from an input text file, maps each reaction
using TemplateReactionMapper + IdenticalFragmentMapper, and saves results as
batch pickle files in the specified output directory.

Each output dict has the structure::

    {
        "rxn": str,                     # original reaction SMILES
        "selected_mapping": str,        # atom-mapped SMILES (empty string if failed)
        "possible_templates": List[str] # matched template names for the mapping
    }

Batch files are named ``batch_000000.pkl``, ``batch_000001.pkl``, ... and each
contains a ``List[dict]`` of up to ``--batch-size`` results.

Example usage::

    python scripts/map_reactions.py \\
        --input  model_training/data/agave_chem_cleaned_data.txt \\
        --output-dir model_training/data/mapped_batches \\
        --workers 8 \\
        --batch-size 10000
"""

import argparse
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import List, Optional

from agave_chem.mappers.identical_fragments.identical_fragment_mapper import (
    IdenticalFragmentMapper,
)
from agave_chem.mappers.template.template_mapper import TemplateReactionMapper
from agave_chem.utils.logging_config import disable_library_logging, logger

try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False


# ── Worker globals ───────────────────────────────────────────────────────────

_template_mapper: Optional[TemplateReactionMapper] = None
_identical_fragment_mapper: Optional[IdenticalFragmentMapper] = None


def _init_worker() -> None:
    """
    Initialize per-worker mapper instances.

    Called exactly once per worker process by ``multiprocessing.Pool``.
    Stores mapper objects in module-level globals so they are reused across
    all tasks handled by this worker.
    """
    global _template_mapper, _identical_fragment_mapper
    disable_library_logging()
    _template_mapper = TemplateReactionMapper("template_default")
    _identical_fragment_mapper = IdenticalFragmentMapper("identical_fragment_helper")


def _map_one(rxn: str) -> dict:
    """
    Atom-map a single reaction SMILES string.

    Uses the module-level ``_template_mapper`` and ``_identical_fragment_mapper``
    initialised by ``_init_worker``. Returns a result dict regardless of success
    or failure; failed reactions get an empty string for ``selected_mapping``.

    Args:
        rxn (str): Reaction SMILES string to map.

    Returns:
        dict:
            - "rxn" (str): The original reaction SMILES.
            - "selected_mapping" (str): Atom-mapped reaction SMILES, or ``""``
              on failure.
            - "possible_templates" (List[str]): Template names matched for the
              selected mapping.
    """
    try:
        rxn_list, identical_fragments_mapping_list = (
            _identical_fragment_mapper.create_identical_fragments_mapping_list([rxn])  # type: ignore
        )

        out_template, out_mcs = _template_mapper.map_reaction_with_mcs_optimization(  # type: ignore
            rxn_list[0]
        )
        return_val = out_template if out_template.get("selected_mapping") else out_mcs

        new_out = _identical_fragment_mapper.resolve_identical_fragments_mapping_list(  # type: ignore
            [return_val["selected_mapping"]],
            [identical_fragments_mapping_list[0]],
        )
        return_val["selected_mapping"] = new_out[0]

        possible_templates: List[str] = []
        for k, v in return_val["possible_mappings"].items():
            if k == return_val["selected_mapping"]:
                for ele in v:
                    possible_templates.append(ele["template_name"])

        return {
            "rxn": rxn,
            "selected_mapping": return_val["selected_mapping"],
            "possible_templates": possible_templates,
        }
    except Exception:
        return {"rxn": rxn, "selected_mapping": "", "possible_templates": []}


# ── I/O helpers ──────────────────────────────────────────────────────────────


def _load_reactions(input_path: Path) -> List[str]:
    """
    Load reaction SMILES from a plain-text file.

    Args:
        input_path (Path): Path to a text file with one reaction SMILES per line.

    Returns:
        List[str]: Non-empty stripped lines from the file.
    """
    with open(input_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _save_batch(batch: List[dict], output_dir: Path, batch_idx: int) -> None:
    """
    Pickle a batch of result dicts to ``output_dir/batch_<idx>.pkl``.

    Args:
        batch (List[dict]): List of result dicts to serialise.
        output_dir (Path): Directory to write the batch file into.
        batch_idx (int): Zero-based batch index used in the filename.
    """
    batch_path = output_dir / f"batch_{batch_idx:06d}.pkl"
    with open(batch_path, "wb") as f:
        pickle.dump(batch, f)
    logger.info(f"Saved batch {batch_idx:06d} ({len(batch)} results) → {batch_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel reaction atom-mapping using TemplateReactionMapper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input .txt file with one reaction SMILES per line.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to write batch pickle files into (created if absent).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Number of results to accumulate before writing a batch pickle file.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=50,
        help="Number of reactions sent to each worker per imap_unordered chunk.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    disable_library_logging()

    logger.info(f"Loading reactions from {args.input} ...")
    rxns = _load_reactions(args.input)
    logger.info(f"Loaded {len(rxns):,} reactions.")
    logger.info(
        f"Starting pool: workers={args.workers}, "
        f"batch_size={args.batch_size:,}, chunksize={args.chunksize}"
    )

    start = time.time()
    batch_idx = 0
    current_batch: List[dict] = []
    processed = 0
    failed = 0

    with mp.Pool(processes=args.workers, initializer=_init_worker) as pool:
        result_iter = pool.imap_unordered(_map_one, rxns, chunksize=args.chunksize)

        if _TQDM_AVAILABLE:
            result_iter = tqdm(result_iter, total=len(rxns), desc="Mapping", unit="rxn")

        for result in result_iter:
            current_batch.append(result)
            processed += 1
            if not result["selected_mapping"]:
                failed += 1

            if len(current_batch) >= args.batch_size:
                _save_batch(current_batch, args.output_dir, batch_idx)
                batch_idx += 1
                current_batch = []

    if current_batch:
        _save_batch(current_batch, args.output_dir, batch_idx)

    elapsed = time.time() - start
    success = processed - failed
    logger.info(
        f"Finished {processed:,} reactions in {elapsed:.1f}s "
        f"({processed / elapsed:.1f} rxn/s) — "
        f"mapped: {success:,} ({100 * success / processed:.1f}%), "
        f"failed: {failed:,} ({100 * failed / processed:.1f}%)."
    )
    logger.info(f"Batch files written to: {args.output_dir}/")


if __name__ == "__main__":
    main()
