import random

def create_comparison_set(batch_images, buffer_images, candidate_image, k):
    """Build a shuffled K-way comparison set.

    Returns (comparison_set, buffer_indices, candidate_index) where buffer_indices
    maps comparison-set position -> original buffer index, and candidate_index is the
    candidate's position in the shuffled set.
    """
    total = len(batch_images) + len(buffer_images)
    n_draw = k - 1

    if n_draw > total:
        raise ValueError(f"Cannot draw {n_draw} images from pool of {total}")

    pool = [("batch", i, img) for i, img in enumerate(batch_images)] + \
           [("buffer", i, img) for i, img in enumerate(buffer_images)]

    drawn = random.sample(pool, n_draw)

    comparison = [("candidate", None, candidate_image)] + drawn
    random.shuffle(comparison)

    comparison_set  = [img for _, _, img in comparison]
    buffer_indices  = {i: src_idx for i, (tag, src_idx, _) in enumerate(comparison) if tag == "buffer"}
    candidate_index = next(i for i, (tag, _, _) in enumerate(comparison) if tag == "candidate")

    return comparison_set, buffer_indices, candidate_index