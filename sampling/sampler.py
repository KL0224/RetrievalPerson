def sample_best_per_window(tracklet_frames, window=30, ratio_to_largest_area=0.5, number_to_aggregate=3):
    sampled = []
    largest_area = 0.0
    for i in range(0, len(tracklet_frames), window):
        chunk = tracklet_frames[i:i+window] # totally safe nha Kiet Lac :)))
        # highest conf one from the window
        best = max(chunk, key=lambda f: f.confidence)

        # compute it's area
        x1, y1, x2, y2 = best.bbox
        area = (x2-x1)*(y2-y1)

        #update largest_area
        largest_area = max(largest_area, area)
        sampled.append((best, area))

    # only keep those that are large enough
    filtered = [item[0] for item in sampled if item[1] >= ratio_to_largest_area*largest_area]
    filtered.sort(key=lambda f: f.confidence, reverse=True)
    return filtered[:number_to_aggregate]

