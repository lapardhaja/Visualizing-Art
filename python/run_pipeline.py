#!/usr/bin/env python3
"""
Complete pipeline to generate CLIP embeddings and create similarity search system.
Run this script to execute the entire pipeline from start to finish.
"""

import sys
import logging
from pathlib import Path
import argparse
import subprocess
import os

from config import METADATA_CONFIG, PIPELINE_CONFIG, EMBEDDINGS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_and_install_dependencies():
    """Check for missing packages and install them automatically."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.warning(f"requirements.txt not found at {requirements_file}")
        return
    
    logger.info("Checking for missing dependencies...")
    
    # Map of import names to package names (for packages with different import names)
    package_map = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'open_clip': 'open-clip-torch',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'PIL': 'Pillow',
        'requests': 'requests',
        'tqdm': 'tqdm',
        'openpyxl': 'openpyxl',
        'xlrd': 'xlrd',
    }
    
    # Check which packages are missing
    missing_packages = []
    for import_name, package_name in package_map.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
            logger.info(f"  [X] Missing: {package_name}")
    
    if not missing_packages:
        logger.info("[OK] All required packages are already installed")
        return
    
    logger.info(f"\nFound {len(missing_packages)} missing package(s): {', '.join(missing_packages)}")
    logger.info("Installing missing packages...")
    
    try:
        # Install from requirements.txt (pip will handle version constraints)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            capture_output=False,  # Show output to user
            text=True
        )
        
        if result.returncode == 0:
            logger.info("[OK] All missing packages installed successfully")
            
            # Verify installation
            logger.info("Verifying installation...")
            all_installed = True
            for import_name, package_name in package_map.items():
                try:
                    __import__(import_name)
                except ImportError:
                    logger.error(f"  [X] {package_name} still not available after installation")
                    all_installed = False
            
            if all_installed:
                logger.info("[OK] All packages verified and ready to use")
            else:
                logger.warning("Some packages may need manual installation")
                logger.warning(f"Run: pip install -r {requirements_file}")
        else:
            logger.error("Failed to install some packages")
            logger.error(f"Please run manually: pip install -r {requirements_file}")
            raise RuntimeError("Dependency installation failed")
            
    except Exception as e:
        logger.error(f"Error during package installation: {e}")
        logger.error(f"Please install manually: pip install -r {requirements_file}")
        raise

def run_pipeline(unknown_strategy: str = None,
                visual_weight: float = None,
                metadata_weight: float = None,
                force_regenerate: bool = False,
                skip_download: bool = False):
    """Run the complete pipeline with configurable options.
    
    Args:
        unknown_strategy: How to handle unknown values in metadata
        visual_weight: Weight for visual features in combination
        metadata_weight: Weight for metadata features in combination
        force_regenerate: Force regeneration of all features
        skip_download: Skip downloading images and metadata
    """
    logger.info("Starting Enhanced CLIP Artwork Embeddings Pipeline")
    logger.info(f"Configuration:")
    logger.info(f"  Unknown handling: {unknown_strategy or METADATA_CONFIG.UNKNOWN_HANDLING_STRATEGY}")
    logger.info(f"  Visual weight: {visual_weight or METADATA_CONFIG.VISUAL_WEIGHT}")
    logger.info(f"  Metadata weight: {metadata_weight or METADATA_CONFIG.METADATA_WEIGHT}")
    
    try:
        # Step 0: Download images and metadata (if needed)
        if not skip_download:
            images_dir = Path("../images")
            metadata_csv = Path("../metadata/artwork_metadata.csv")
            
            # Check if we need to download
            needs_download = force_regenerate or not images_dir.exists() or not list(images_dir.glob("*.jpg")) or not metadata_csv.exists()
            
            if needs_download:
                logger.info("Step 0: Downloading images and metadata from Met Museum API")
                try:
                    from download_met_images import choose_data_file, process_artworks
                    data_file = choose_data_file()
                    process_artworks(data_file, limit=None)
                    logger.info("Images and metadata downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download images: {e}")
                    logger.error("You can skip this step with --skip-download if images are already available")
                    raise
            else:
                logger.info("Step 0: Images and metadata already exist, skipping download")
        else:
            logger.info("Step 0: Skipping download (--skip-download flag set)")
        
        # Step 1: Generate CLIP embeddings (if needed)
        clip_path = EMBEDDINGS_DIR / "clip_embeddings.npy"
        if force_regenerate or not clip_path.exists():
            logger.info("Step 1: Generating CLIP embeddings")
            from generate_embeddings import generate_embeddings
            generate_embeddings()
            logger.info("CLIP embeddings generated successfully")
        else:
            logger.info("Step 1: CLIP embeddings already exist, skipping")
        
        # Step 2: Process metadata with new unknown handling
        metadata_path = EMBEDDINGS_DIR / "metadata_features.npy"
        if force_regenerate or not metadata_path.exists() or unknown_strategy:
            logger.info("Step 2: Processing metadata features")
            from process_metadata import process_metadata
            process_metadata(unknown_strategy=unknown_strategy)
            logger.info("Metadata features processed successfully")
        else:
            logger.info("Step 2: Metadata features already exist, skipping")
        
        # Step 3: Combine features with weighting
        combined_path = EMBEDDINGS_DIR / "combined_embeddings.npy"
        if force_regenerate or not combined_path.exists() or visual_weight or metadata_weight:
            logger.info("Step 3: Combining CLIP and metadata features")
            from combine_features import combine_features
            combine_features(visual_weight=visual_weight, metadata_weight=metadata_weight)
            logger.info("Features combined successfully")
        else:
            logger.info("Step 3: Combined features already exist, skipping")
        
        # Step 4: Generate enhanced similarities
        logger.info("Step 4: Generating enhanced similarities")
        
        # Generate similarity modes for summary_combined.csv and summary_cross_artist.csv
        similarity_modes = [
            ["--mode", "combined", "--top-k", "5"],  # Creates summary_combined.csv
            ["--mode", "combined", "--exclude-same-artist", "--top-k", "5", "--output-suffix", "cross_artist"]  # Creates summary_cross_artist.csv
        ]
        
        for mode_args in similarity_modes:
            logger.info(f"Generating similarities with args: {' '.join(mode_args)}")
            try:
                subprocess.run([sys.executable, "generate_all_similarities.py"] + mode_args, check=True)
                logger.info(f"Successfully generated similarities for: {' '.join(mode_args)}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to generate similarities for {' '.join(mode_args)}: {e}")
        
        # Step 5: Add metadata to summary_cross_artist.csv
        logger.info("Step 5: Adding metadata to similarity results")
        try:
            from add_metadata import add_metadata
            add_metadata()
            logger.info("Metadata added successfully")
        except Exception as e:
            logger.error(f"Failed to add metadata: {e}")
            raise
        
        logger.info("=" * 70)
        logger.info("[OK] Pipeline completed successfully!")
        logger.info("=" * 70)
        logger.info("Enhanced similarity system is ready with zero-vector unknown handling!")
        logger.info("")
        logger.info("Key improvements:")
        logger.info("  [OK] Unknown values no longer create false similarities")
        logger.info("  [OK] Adaptive weighting based on metadata completeness")
        logger.info("  [OK] Configurable visual/metadata balance")
        logger.info("")
        logger.info("Generated similarity files:")
        logger.info("  - summary_combined.csv (CLIP + metadata similarities)")
        logger.info("  - summary_cross_artist.csv (combined mode, excluding same artist)")
        logger.info("")
        logger.info("Files saved in ../embeddings/ directory")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Run the CLIP artwork embeddings pipeline with enhanced configuration'
    )
    parser.add_argument('--unknown-strategy', 
                       choices=['zero_vector', 'category', 'missing_indicator'],
                       default=None,
                       help='How to handle unknown/missing categorical values')
    parser.add_argument('--visual-weight', 
                       type=float, 
                       default=None,
                       help='Weight for visual features (default: from config)')
    parser.add_argument('--metadata-weight', 
                       type=float, 
                       default=None,
                       help='Weight for metadata features (default: from config)')
    parser.add_argument('--force-regenerate', 
                       action='store_true',
                       help='Force regeneration of all features')
    parser.add_argument('--skip-download', 
                       action='store_true',
                       help='Skip downloading images and metadata (use if already downloaded)')
    parser.add_argument('--skip-deps-check', 
                       action='store_true',
                       help='Skip dependency checking and installation')
    parser.add_argument('--no-server', 
                       action='store_true',
                       help='Skip starting local HTTP server after pipeline completes')
    
    args = parser.parse_args()
    
    # Check and install dependencies first
    if not args.skip_deps_check:
        check_and_install_dependencies()
    
    run_pipeline(
        unknown_strategy=args.unknown_strategy,
        visual_weight=args.visual_weight,
        metadata_weight=args.metadata_weight,
        force_regenerate=args.force_regenerate,
        skip_download=args.skip_download
    )
    
    # Start HTTP server automatically (unless --no-server flag is used)
    if not args.no_server:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Starting local HTTP server on port 5500...")
        logger.info("Server will be available at: http://localhost:5500")
        logger.info("Press Ctrl+C to stop the server")
        logger.info("=" * 70)
        try:
            # Change to project root directory (one level up from python/)
            project_root = Path(__file__).parent.parent
            os.chdir(project_root)
            subprocess.run([sys.executable, "-m", "http.server", "5500"])
        except KeyboardInterrupt:
            logger.info("HTTP server stopped by user")
        except Exception as e:
            logger.error(f"Error starting HTTP server: {e}")

if __name__ == "__main__":
    main()
