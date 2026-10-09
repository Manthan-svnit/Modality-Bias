"""
End-to-end orchestrator for the Multimodal RCS pipeline.

Runs all 3 stages sequentially for both Early and Late fusion variants
on any dataset with any combination of active modalities.
"""

import os
import argparse
import logging
import time

from multimodal_rcs.config import PipelineConfig, add_pipeline_args, config_from_args
from train_stage1 import train_stage1
from train_stage2 import run_stage2
from train_stage3 import run_stage3

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EndToEndPipeline")


def run_full_pipeline(config: PipelineConfig):
    start_time = time.time()
    logger.info("=" * 80)
    logger.info(
        f"STARTING END-TO-END MULTIMODAL RCS PIPELINE | "
        f"Dataset: {config.dataset_name} | "
        f"Modalities: {config.active_modalities}"
    )
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # 1. STAGE 1: EARLY FUSION
    # ------------------------------------------------------------------
    logger.info("\n>>> [STEP 1/5] Stage 1: Early Fusion Training...")
    config.fusion_variant = "early"
    train_stage1(config)

    # ------------------------------------------------------------------
    # 2. STAGE 1: LATE FUSION
    # ------------------------------------------------------------------
    logger.info("\n>>> [STEP 2/5] Stage 1: Late Fusion Training...")
    config.fusion_variant = "late"
    train_stage1(config)

    # ------------------------------------------------------------------
    # 3. STAGE 2: FEATURE EXTRACTION + RCS — EARLY FUSION
    # ------------------------------------------------------------------
    logger.info("\n>>> [STEP 3/5] Stage 2: RCS Balancing (Early Fusion)...")
    config.fusion_variant = "early"
    run_stage2(config)

    # ------------------------------------------------------------------
    # 4. STAGE 2: FEATURE EXTRACTION + RCS — LATE FUSION
    # ------------------------------------------------------------------
    logger.info("\n>>> [STEP 4/5] Stage 2: RCS Balancing (Late Fusion)...")
    config.fusion_variant = "late"
    run_stage2(config)

    # ------------------------------------------------------------------
    # 5. STAGE 3: CLASSIFIER TRAINING + BENCHMARK (both variants)
    # ------------------------------------------------------------------
    logger.info("\n>>> [STEP 5/5] Stage 3: Classifier Training and Benchmark Evaluation...")
    run_stage3(config)

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"PIPELINE COMPLETE in {elapsed / 60:.1f} min.")
    logger.info(
        f"Results saved under: results/{config.dataset_name}/ and "
        f"checkpoints/{config.dataset_name}/"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-End Multimodal RCS Pipeline (all stages)"
    )
    parser = add_pipeline_args(parser)
    args = parser.parse_args()
    cfg = config_from_args(args)
    run_full_pipeline(cfg)
