import os
import argparse
import logging
import time

from multimodal_rcs.config import PipelineConfig, add_pipeline_args
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
    logger.info("STARTING END-TO-END MULTIMODAL RCS RESEARCH PIPELINE")
    logger.info("=" * 80)

    # ---------------------------------------------------------
    # 1. STAGE 1: EARLY FUSION
    # ---------------------------------------------------------
    logger.info("\n>>> [STEP 1/5] Training Stage 1: Variant 1 (Early Fusion)...")
    config.fusion_variant = "early"
    train_stage1(config)

    # ---------------------------------------------------------
    # 2. STAGE 1: LATE FUSION
    # ---------------------------------------------------------
    logger.info("\n>>> [STEP 2/5] Training Stage 1: Variant 2 (Late Fusion)...")
    config.fusion_variant = "late"
    train_stage1(config)

    # ---------------------------------------------------------
    # 3. STAGE 2: RCS BALANCING FOR EARLY FUSION
    # ---------------------------------------------------------
    logger.info("\n>>> [STEP 3/5] Stage 2: Feature Extraction & RCS Balancing (Early Fusion)...")
    config.fusion_variant = "early"
    run_stage2(config)

    # ---------------------------------------------------------
    # 4. STAGE 2: RCS BALANCING FOR LATE FUSION
    # ---------------------------------------------------------
    logger.info("\n>>> [STEP 4/5] Stage 2: Feature Extraction & RCS Balancing (Late Fusion)...")
    config.fusion_variant = "late"
    run_stage2(config)

    # ---------------------------------------------------------
    # 5. STAGE 3: CLASSIFIER & BENCHMARK EVALUATION (4 VARIANTS + BASELINES)
    # ---------------------------------------------------------
    logger.info("\n>>> [STEP 5/5] Stage 3: Classifier Training & Comprehensive Benchmark Evaluation...")
    results = run_stage3(config)

    elapsed = time.time() - start_time
    logger.info("=" * 80)
    logger.info(f"MULTIMODAL RCS PIPELINE FINISHED SUCCESSFULLY IN {elapsed / 60:.2f} MINUTES!")
    logger.info(f"Check results at: '{config.results_dir}/benchmark_results.md'")
    logger.info(f"Figures saved in: '{config.figures_dir}/'")
    logger.info("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Multimodal RCS Orchestrator")
    parser = add_pipeline_args(parser)
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        device=args.device,
        stage1_epochs=args.stage1_epochs,
        stage1_lr=args.stage1_lr,
        batch_size=args.batch_size,
        eta=args.eta,
        shrinkage_lambda=args.shrinkage_lambda,
        clf_epochs=args.clf_epochs,
        clf_lr=args.clf_lr,
        random_seed=args.random_seed,
    )
    run_full_pipeline(cfg)
