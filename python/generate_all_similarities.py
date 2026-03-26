import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import logging
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_DIR = ROOT / "embeddings"
DATA_DIR = ROOT / "data"
METADATA_CSV = ROOT / "metadata" / "artwork_metadata.csv"

def load_data(mode='combined'):
    logger.info(f"Loading data for mode: {mode}")

    if mode == 'visual_only':
        embeddings = np.load(EMBEDDINGS_DIR / "clip_embeddings.npy")
        object_ids = np.load(EMBEDDINGS_DIR / "object_ids.npy")
    elif mode == 'combined':
        embeddings = np.load(EMBEDDINGS_DIR / "combined_embeddings.npy")
        object_ids = np.load(EMBEDDINGS_DIR / "combined_object_ids.npy")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metadata = pd.read_csv(METADATA_CSV)

    logger.info(f"Loaded {len(object_ids)} embeddings, dim={embeddings.shape[1]}")
    return embeddings, object_ids, metadata

def calculate_similarity_matrix(embeddings):
    logger.info("Calculating cosine similarity matrix")
    sim = cosine_similarity(embeddings)
    np.fill_diagonal(sim, 0)
    return sim

def find_top_similarities(sim_mat, object_ids, metadata, top_k=3,
                          exclude_same_artist=False,
                          exclude_same_medium=False,
                          min_similarity_threshold=0.0):

    logger.info("Precomputing artist/medium maps...")
    artist_map = {}
    medium_map = {}
    for i, oid in enumerate(object_ids):
        row = metadata[metadata['Object ID'] == oid]
        if not row.empty:
            artist_map[oid] = row.iloc[0].get('Artist Display Name', '')
            medium_map[oid] = row.iloc[0].get('Medium', '')
        else:
            artist_map[oid] = ''
            medium_map[oid] = ''

    results = {}

    for i, oid in enumerate(tqdm(object_ids, desc="Computing similarities")):
        row = metadata[metadata['Object ID'] == oid]
        if row.empty:
            continue

        query_info = {
            'object_id': int(oid),
            'title': row.iloc[0].get('Title', ''),
            'artist': row.iloc[0].get('Artist Display Name', ''),
            'medium': row.iloc[0].get('Medium', ''),
            'department': row.iloc[0].get('Department', ''),
            'object_date': row.iloc[0].get('Object Date', ''),
            'tags': row.iloc[0].get('Tags', '')
        }

        sims = sim_mat[i]
        valid_mask = sims >= min_similarity_threshold

        if exclude_same_artist and query_info['artist']:
            for j, other_oid in enumerate(object_ids):
                if artist_map[other_oid] == query_info['artist']:
                    valid_mask[j] = False

        if exclude_same_medium and query_info['medium']:
            for j, other_oid in enumerate(object_ids):
                if medium_map[other_oid] == query_info['medium']:
                    valid_mask[j] = False

        # gather candidates
        valid_scores = sims[valid_mask]
        valid_idx = np.where(valid_mask)[0]

        if not len(valid_scores):
            continue

        top_k_actual = min(top_k, len(valid_scores))
        order_in_valid = np.argsort(valid_scores)[::-1][:top_k_actual]
        top_idx = valid_idx[order_in_valid]
        top_scores = valid_scores[order_in_valid]
        top_oids = object_ids[top_idx]

        similar_artworks = []
        for rank, (other_oid, score) in enumerate(zip(top_oids, top_scores), start=1):
            other_row = metadata[metadata['Object ID'] == other_oid]
            if other_row.empty:
                continue
            similar_artworks.append({
                'rank': rank,
                'object_id': int(other_oid),
                'similarity_score': float(score),
                'title': other_row.iloc[0].get('Title', ''),
                'artist': other_row.iloc[0].get('Artist Display Name', ''),
                'medium': other_row.iloc[0].get('Medium', ''),
                'department': other_row.iloc[0].get('Department', ''),
                'object_date': other_row.iloc[0].get('Object Date', ''),
                'tags': other_row.iloc[0].get('Tags', '')
            })

        results[int(oid)] = {
            'query_artwork': query_info,
            'top_similar': similar_artworks
        }

    return results

def save_results(results, output_file):
    EMBEDDINGS_DIR.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

def create_summary_csv(results, output_file):
    rows = []
    for oid, rec in results.items():
        q = rec['query_artwork']
        for sim in rec['top_similar']:
            rows.append({
                'query_object_id': oid,
                'query_title': q['title'],
                'query_artist': q['artist'],
                'query_medium': q['medium'],
                'similar_object_id': sim['object_id'],
                'similar_title': sim['title'],
                'similar_artist': sim['artist'],
                'similar_medium': sim['medium'],
                'similarity_score': sim['similarity_score'],
                'rank': sim['rank']
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

def generate_statistics(results):
    all_scores = []
    artist_matches = 0
    medium_matches = 0
    total_pairs = 0
    cross_artist_pairs = 0

    for oid, rec in results.items():
        q = rec['query_artwork']
        for sim in rec['top_similar']:
            all_scores.append(sim['similarity_score'])
            total_pairs += 1
            if q['artist'] and q['artist'] == sim['artist']:
                artist_matches += 1
            else:
                cross_artist_pairs += 1
            if q['medium'] and q['medium'] == sim['medium']:
                medium_matches += 1

    stats = {
        "total_artworks": len(results),
        "total_similarity_pairs": total_pairs,
        "score_statistics": {
            "min": float(np.min(all_scores)) if all_scores else 0,
            "max": float(np.max(all_scores)) if all_scores else 0,
            "mean": float(np.mean(all_scores)) if all_scores else 0,
            "std": float(np.std(all_scores)) if all_scores else 0,
            "median": float(np.median(all_scores)) if all_scores else 0
        },
        "matching_statistics": {
            "artist_matches": artist_matches,
            "artist_match_rate": artist_matches/total_pairs if total_pairs else 0,
            "cross_artist_pairs": cross_artist_pairs,
            "cross_artist_rate": cross_artist_pairs/total_pairs if total_pairs else 0,
            "medium_matches": medium_matches,
            "medium_match_rate": medium_matches/total_pairs if total_pairs else 0
        }
    }
    return stats

def main():
    parser = argparse.ArgumentParser(description="Generate artwork similarities")
    parser.add_argument('--mode', choices=['combined','visual_only'], default='combined')
    parser.add_argument('--exclude-same-artist', action='store_true')
    parser.add_argument('--exclude-same-medium', action='store_true')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--min-threshold', type=float, default=0.0)
    parser.add_argument('--output-suffix', type=str, default='')
    args = parser.parse_args()

    embeddings, object_ids, metadata = load_data(args.mode)
    sim_mat = calculate_similarity_matrix(embeddings)

    results = find_top_similarities(
        sim_mat, object_ids, metadata,
        top_k=args.top_k,
        exclude_same_artist=args.exclude_same_artist,
        exclude_same_medium=args.exclude_same_medium,
        min_similarity_threshold=args.min_threshold
    )

    # suffix naming
    if args.output_suffix:
        suffix = "_" + args.output_suffix
    else:
        parts = []
        if args.mode == 'visual_only':
            parts.append("visual")
        if args.exclude_same_artist:
            parts.append("cross_artist")
        if args.exclude_same_medium:
            parts.append("cross_medium")
        if args.min_threshold > 0:
            parts.append(f"min{args.min_threshold:.2f}")
        suffix = "_" + "_".join(parts) if parts else "_combined"

    out_json = EMBEDDINGS_DIR / f"similarities{suffix}.json"
    out_csv = DATA_DIR / f"summary{suffix}.csv"
    out_stats = EMBEDDINGS_DIR / f"statistics{suffix}.json"

    save_results(results, out_json)
    create_summary_csv(results, out_csv)
    stats = generate_statistics(results)
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    # Log some quick samples
    example_ids = list(results.keys())[:3]
    for oid in example_ids:
        rec = results[oid]
        q = rec['query_artwork']
        logger.info(f"\nObject {oid}: {q['title']} by {q['artist']}")
        for sim in rec['top_similar']:
            logger.info(f"  {sim['rank']}. {sim['title']} by {sim['artist']} (Score: {sim['similarity_score']:.4f})")

if __name__ == "__main__":
    main()
