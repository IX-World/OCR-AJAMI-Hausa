"""CTC prefix beam search in log space, without a lexicon or language model."""
import math


NEG_INF = -math.inf


def logadd(a, b):
    if a < b:
        a, b = b, a
    return a if b == NEG_INF else a + math.log1p(math.exp(b - a))


def prefix_beam_search(log_probs, beam_width=25, blank=0, nbest=5, token_top_k=0):
    """Return (token tuple, log CTC probability) hypotheses, best first.

    Scores sum alignment probabilities for each prefix. Pruning the beam is
    approximate; token_top_k=0 retains every label at every frame. No length
    bonus, reference text, linguistic score, or post-processing is applied.
    """
    if beam_width < 1 or nbest < 1 or token_top_k < 0:
        raise ValueError("Beam, nbest et token-top-k invalides.")
    beam = {(): (0.0, NEG_INF)}
    classes = None
    for frame in log_probs:
        frame = list(frame)
        if classes is None:
            classes = len(frame)
            if not 0 <= blank < classes:
                raise ValueError("Indice BLANK invalide.")
        if len(frame) != classes or any(math.isnan(p) or p > 1e-5 for p in frame) or max(frame) == NEG_INF:
            raise ValueError("Probabilites logarithmiques invalides.")
        candidates = [i for i in range(classes) if i != blank and frame[i] != NEG_INF]
        if token_top_k:
            candidates = sorted(candidates, key=lambda i: (-frame[i], i))[:token_top_k]
        following = {}
        for prefix, (p_blank, p_nonblank) in beam.items():
            total = logadd(p_blank, p_nonblank)
            same = following.setdefault(prefix, [NEG_INF, NEG_INF])
            same[0] = logadd(same[0], total + frame[blank])
            last = prefix[-1] if prefix else None
            labels = candidates
            if token_top_k and last is not None and last not in labels and frame[last] != NEG_INF:
                labels = candidates + [last]
            for label in labels:
                probability = frame[label]
                if label == last:
                    # A repeat without BLANK keeps the prefix; after BLANK it
                    # produces a second occurrence of the same character.
                    same[1] = logadd(same[1], p_nonblank + probability)
                    extension_probability = p_blank + probability
                else:
                    extension_probability = total + probability
                if extension_probability != NEG_INF:
                    extended = following.setdefault(prefix + (label,), [NEG_INF, NEG_INF])
                    extended[1] = logadd(extended[1], extension_probability)
        ranked = sorted(((prefix, values) for prefix, values in following.items() if logadd(*values) != NEG_INF), key=lambda item: (-logadd(*item[1]), item[0]))
        beam = dict(ranked[:beam_width])
    ranked = sorted(((prefix, logadd(*score)) for prefix, score in beam.items()), key=lambda item: (-item[1], item[0]))
    return ranked[:nbest]


def tokens_to_text(tokens, vocab):
    inverse = {index: char for char, index in vocab.items()}
    # Prefix search has already removed CTC repeats; do NOT collapse again.
    return "".join("\ufffd" if inverse[i] == "<UNK>" else inverse[i] for i in tokens)
