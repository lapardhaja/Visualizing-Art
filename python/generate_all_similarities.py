import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
EMBEDDINGS_DIR = Path("../embeddings")
METADATA_CSV = Path("../metadata/artwork_metadata.csv")

def load_data(mode='combined'):
    """Load embeddings and metadata based on mode."""
    logger.info(f"Loading data for mode: {mode}")
    
    if mode == 'visual_only':
        # Load pure CLIP embeddings for visual-only similarity
        embeddings = np.load(EMBEDDINGS_DIR / "clip_embeddings.npy")
        object_ids = np.load(EMBEDDINGS_DIR / "object_ids.npy")
        logger.info(f"Loaded CLIP embeddings: {embeddings.shape}")
        
    elif mode == 'combined':
        # Load combined embeddings (CLIP + metadata)
        embeddings = np.load(EMBEDDINGS_DIR / "combined_embeddings.npy")
        object_ids = np.load(EMBEDDINGS_DIR / "combined_object_ids.npy")
        logger.info(f"Loaded combined embeddings: {embeddings.shape}")
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Load metadata
    metadata = pd.read_csv(METADATA_CSV)
    
    logger.info(f"Loaded {len(object_ids)} embeddings")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    
    return embeddings, object_ids, metadata

def calculate_similarity_matrix(embeddings):
    """Calculate similarity matrix for all artworks."""
    logger.info("Calculating similarity matrix for all artworks")
    
    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Set diagonal to 0 (self-similarity)
    np.fill_diagonal(similarity_matrix, 0)
    
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    return similarity_matrix

def find_top_similarities(similarity_matrix, object_ids, metadata, top_k=3, 
                         exclude_same_artist=False, exclude_same_medium=False, 
                         min_similarity_threshold=0.0):
    """Find top-k most similar artworks with filtering options.
    
    If top_k is None, returns all valid similarities (no limit).
    """
    if top_k is None:
        logger.info("Finding ALL similar artworks for each piece (no limit)")
    else:
        logger.info(f"Finding top {top_k} similar artworks for each piece")
    logger.info(f"Exclude same artist: {exclude_same_artist}")
    logger.info(f"Exclude same medium: {exclude_same_medium}")
    logger.info(f"Min similarity threshold: {min_similarity_threshold}")
    
    # Pre-compute artist, medium, and full metadata mappings for efficiency
    logger.info("Pre-computing metadata mappings...")
    artist_mapping = {}
    medium_mapping = {}
    metadata_mapping = {}
    
    # Build a dictionary from metadata CSV for O(1) lookups
    metadata_dict = metadata.set_index('Object ID').to_dict('index')
    
    for i, obj_id in enumerate(object_ids):
        if obj_id in metadata_dict:
            row = metadata_dict[obj_id]
            artist_mapping[obj_id] = row.get('Artist Display Name', '')
            medium_mapping[obj_id] = row.get('Medium', '')
            metadata_mapping[obj_id] = row
        else:
            artist_mapping[obj_id] = ''
            medium_mapping[obj_id] = ''
            metadata_mapping[obj_id] = {}
    
    logger.info("Mappings computed, starting similarity analysis...")
    
    results = {}
    
    for i, obj_id in enumerate(tqdm(object_ids, desc="Processing artworks")):
        # Get similarity scores for this artwork
        similarities = similarity_matrix[i]
        
        # Get metadata for query artwork (using pre-computed mapping)
        query_meta = metadata_mapping.get(obj_id, {})
        if not query_meta:
            continue
            
        query_info = {
            'object_id': int(obj_id),
            'title': query_meta.get('Title', ''),
            'artist': query_meta.get('Artist Display Name', ''),
            'medium': query_meta.get('Medium', ''),
            'department': query_meta.get('Department', ''),
            'object_date': query_meta.get('Object Date', ''),
            'object_begin_date': query_meta.get('Object Begin Date', ''),
            'object_end_date': query_meta.get('Object End Date', ''),
            'tags': query_meta.get('Tags', '')
        }
        
        # Create mask for filtering
        valid_indices = np.ones(len(similarities), dtype=bool)
        
        # Apply similarity threshold
        valid_indices &= similarities >= min_similarity_threshold
        
        # Apply artist filtering (now efficient!)
        if exclude_same_artist and query_info['artist']:
            for j, other_obj_id in enumerate(object_ids):
                if artist_mapping[other_obj_id] == query_info['artist']:
                    valid_indices[j] = False
        
        # Apply medium filtering (now efficient!)
        if exclude_same_medium and query_info['medium']:
            for j, other_obj_id in enumerate(object_ids):
                if medium_mapping[other_obj_id] == query_info['medium']:
                    valid_indices[j] = False
        
        # Get valid similarities and indices
        valid_similarities = similarities[valid_indices]
        valid_indices_array = np.where(valid_indices)[0]
        
        if len(valid_similarities) == 0:
            # Provide diagnostic information
            max_similarity = np.max(similarities)
            n_above_threshold = np.sum(similarities >= min_similarity_threshold)
            n_same_artist = 0
            if exclude_same_artist and query_info['artist']:
                n_same_artist = sum(1 for other_id in object_ids 
                                  if artist_mapping.get(other_id) == query_info['artist'])
            
            reason = []
            if max_similarity < min_similarity_threshold:
                reason.append(f"max similarity {max_similarity:.3f} < threshold {min_similarity_threshold}")
            if exclude_same_artist and n_same_artist > 0:
                reason.append(f"{n_same_artist} same-artist matches excluded")
            if exclude_same_medium and query_info['medium']:
                n_same_medium = sum(1 for other_id in object_ids 
                                   if medium_mapping.get(other_id) == query_info['medium'])
                if n_same_medium > 0:
                    reason.append(f"{n_same_medium} same-medium matches excluded")
            
            reason_str = "; ".join(reason) if reason else "all similarities filtered out"
            # Handle NaN or non-string titles
            title_str = str(query_info.get('title', 'Unknown'))[:50] if pd.notna(query_info.get('title')) else 'Unknown'
            logger.warning(f"No valid similarities found for object {obj_id} "
                         f"(title: {title_str}). "
                         f"Reason: {reason_str}. "
                         f"Max similarity: {max_similarity:.3f}, "
                         f"Above threshold: {n_above_threshold}")
            continue
        
        # Get top-k indices from valid candidates (or all if top_k is None)
        sorted_indices = np.argsort(valid_similarities)[::-1]  # Sort descending by similarity
        if top_k is None:
            # No limit - use all valid similarities
            top_indices_in_valid = sorted_indices
        else:
            # Limit to top_k
            top_k_actual = min(top_k, len(valid_similarities))
            top_indices_in_valid = sorted_indices[:top_k_actual]
        
        top_indices = valid_indices_array[top_indices_in_valid]
        top_scores = valid_similarities[top_indices_in_valid]
        top_object_ids = object_ids[top_indices]
        
        # Get metadata for similar artworks (using pre-computed mapping)
        similar_artworks = []
        for j, (similar_obj_id, score) in enumerate(zip(top_object_ids, top_scores)):
            similar_meta = metadata_mapping.get(similar_obj_id, {})
            if similar_meta:
                similar_info = {
                    'rank': j + 1,
                    'object_id': int(similar_obj_id),
                    'similarity_score': float(score),
                    'title': similar_meta.get('Title', ''),
                    'artist': similar_meta.get('Artist Display Name', ''),
                    'medium': similar_meta.get('Medium', ''),
                    'department': similar_meta.get('Department', ''),
                    'object_date': similar_meta.get('Object Date', ''),
                    'object_begin_date': similar_meta.get('Object Begin Date', ''),
                    'object_end_date': similar_meta.get('Object End Date', ''),
                    'tags': similar_meta.get('Tags', '')
                }
                similar_artworks.append(similar_info)
        
        results[int(obj_id)] = {
            'query_artwork': query_info,
            'top_similar': similar_artworks
        }
    
    return results

def save_results(results, output_file):
    """Save results to JSON file."""
    logger.info(f"Saving results to {output_file}")
    
    # Create embeddings directory if it doesn't exist
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_file}")

def create_summary_csv(results, output_file):
    """Create a summary CSV with top similarities."""
    logger.info(f"Creating summary CSV: {output_file}")
    
    summary_data = []
    
    for obj_id, data in results.items():
        query = data['query_artwork']
        
        for similar in data['top_similar']:
            summary_data.append({
                'query_object_id': obj_id,
                'query_title': query['title'],
                'query_artist': query['artist'],
                'query_medium': query['medium'],
                'query_object_begin_date': query.get('object_begin_date', ''),
                'query_object_end_date': query.get('object_end_date', ''),
                'similar_object_id': similar['object_id'],
                'similar_title': similar['title'],
                'similar_artist': similar['artist'],
                'similar_medium': similar['medium'],
                'similar_object_begin_date': similar.get('object_begin_date', ''),
                'similar_object_end_date': similar.get('object_end_date', ''),
                'similarity_score': similar['similarity_score'],
                'rank': similar['rank']
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Summary CSV saved to {output_file}")
    logger.info(f"Total similarity pairs: {len(summary_data)}")

def generate_statistics(results):
    """Generate statistics about the similarity results."""
    logger.info("Generating statistics")
    
    all_scores = []
    artist_matches = 0
    medium_matches = 0
    total_pairs = 0
    cross_artist_pairs = 0
    
    for obj_id, data in results.items():
        query = data['query_artwork']
        
        for similar in data['top_similar']:
            all_scores.append(similar['similarity_score'])
            total_pairs += 1
            
            # Check artist match
            if query['artist'] == similar['artist'] and query['artist'] != '':
                artist_matches += 1
            else:
                cross_artist_pairs += 1
            
            # Check medium match
            if query['medium'] == similar['medium'] and query['medium'] != '':
                medium_matches += 1
    
    stats = {
        'total_artworks': len(results),
        'total_similarity_pairs': total_pairs,
        'score_statistics': {
            'min': float(np.min(all_scores)) if all_scores else 0,
            'max': float(np.max(all_scores)) if all_scores else 0,
            'mean': float(np.mean(all_scores)) if all_scores else 0,
            'std': float(np.std(all_scores)) if all_scores else 0,
            'median': float(np.median(all_scores)) if all_scores else 0
        },
        'matching_statistics': {
            'artist_matches': artist_matches,
            'artist_match_rate': artist_matches / total_pairs if total_pairs > 0 else 0,
            'cross_artist_pairs': cross_artist_pairs,
            'cross_artist_rate': cross_artist_pairs / total_pairs if total_pairs > 0 else 0,
            'medium_matches': medium_matches,
            'medium_match_rate': medium_matches / total_pairs if total_pairs > 0 else 0
        }
    }
    
    return stats

def main():
    """Main function to generate similarities with command line options."""
    parser = argparse.ArgumentParser(description='Generate artwork similarities with enhanced options')
    parser.add_argument('--mode', choices=['combined', 'visual_only'], default='combined',
                       help='Similarity mode: combined (CLIP+metadata) or visual_only (CLIP only)')
    parser.add_argument('--exclude-same-artist', action='store_true',
                       help='Exclude artworks from the same artist')
    parser.add_argument('--exclude-same-medium', action='store_true',
                       help='Exclude artworks with the same medium')
    parser.add_argument('--top-k', type=int, default=3,
                       help='Number of top similar artworks to find')
    parser.add_argument('--no-limit', action='store_true',
                       help='Find ALL similar artworks (no limit per art piece)')
    parser.add_argument('--min-threshold', type=float, default=0.0,
                       help='Minimum similarity threshold')
    parser.add_argument('--output-suffix', type=str, default='',
                       help='Suffix for output files')
    
    args = parser.parse_args()
    
    # If no-limit is set, ignore top-k
    top_k_value = None if args.no_limit else args.top_k
    
    logger.info("=" * 60)
    logger.info("ENHANCED SIMILARITY ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Exclude same artist: {args.exclude_same_artist}")
    logger.info(f"Exclude same medium: {args.exclude_same_medium}")
    if args.no_limit:
        logger.info("Top K: ALL (no limit)")
    else:
        logger.info(f"Top K: {args.top_k}")
    logger.info(f"Min similarity threshold: {args.min_threshold}")
    logger.info("=" * 60)
    
    # Load data
    embeddings, object_ids, metadata = load_data(args.mode)
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(embeddings)
    
    # Find top similarities with filtering
    results = find_top_similarities(
        similarity_matrix, object_ids, metadata, 
        top_k=top_k_value,
        exclude_same_artist=args.exclude_same_artist,
        exclude_same_medium=args.exclude_same_medium,
        min_similarity_threshold=args.min_threshold
    )
    
    # Generate clean output filenames based on parameters
    if args.output_suffix:
        # Use custom suffix if provided
        output_suffix = f"_{args.output_suffix}"
    else:
        # Generate descriptive suffix based on parameters
        parts = []
        if args.mode == 'visual_only':
            parts.append("visual")
        if args.no_limit:
            parts.append("no_limit")
        if args.exclude_same_artist:
            parts.append("cross_artist")
        if args.exclude_same_medium:
            parts.append("cross_medium")
        if args.min_threshold > 0:
            parts.append(f"min{args.min_threshold:.2f}")
        
        if parts:
            output_suffix = f"_{'_'.join(parts)}"
        else:
            output_suffix = "_combined"
    
    # Save detailed results
    output_file = EMBEDDINGS_DIR / f"similarities{output_suffix}.json"
    save_results(results, output_file)
    
    # Create summary CSV
    summary_file = EMBEDDINGS_DIR / f"summary{output_suffix}.csv"
    create_summary_csv(results, summary_file)
    
    # Generate and save statistics
    stats = generate_statistics(results)
    stats_file = EMBEDDINGS_DIR / f"statistics{output_suffix}.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("SIMILARITY ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total artworks analyzed: {stats['total_artworks']}")
    logger.info(f"Total similarity pairs: {stats['total_similarity_pairs']}")
    logger.info(f"Score range: {stats['score_statistics']['min']:.4f} - {stats['score_statistics']['max']:.4f}")
    logger.info(f"Average score: {stats['score_statistics']['mean']:.4f}")
    logger.info(f"Artist match rate: {stats['matching_statistics']['artist_match_rate']:.2%}")
    logger.info(f"Cross-artist rate: {stats['matching_statistics']['cross_artist_rate']:.2%}")
    logger.info(f"Medium match rate: {stats['matching_statistics']['medium_match_rate']:.2%}")
    logger.info("=" * 60)
    
    # Show some examples
    logger.info("Example results:")
    example_ids = list(results.keys())[:3]
    for obj_id in example_ids:
        data = results[obj_id]
        query = data['query_artwork']
        logger.info(f"\nObject {obj_id}: {query['title']} by {query['artist']}")
        for similar in data['top_similar']:
            logger.info(f"  {similar['rank']}. {similar['title']} by {similar['artist']} (Score: {similar['similarity_score']:.4f})")

if __name__ == "__main__":
    main()
