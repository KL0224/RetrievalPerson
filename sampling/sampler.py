def sample_best_per_window(frames, window=10):
    sampled = []
    for i in range(0, len(frames), window):
        chunk = frames[i:i+window]
        best = max(chunk, key=lambda f: f.confidence)
        sampled.append(best)
    return sampled



# 1s -> 1 frame
# 10s -> 10 frames

# 3 3 3 1


def group_seconds(frames, seconds=3):
    step = seconds
    return [
        frames[i:i+step]
        for i in range(0, len(frames), step)
    ]

