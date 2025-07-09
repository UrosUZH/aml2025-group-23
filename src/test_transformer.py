from Transformer import TransformerDecoder


# Example candidate lists for testing the TransformerDecoder

candidate_lists = [
    [("HELLO", 0.22), ("HI", 0.21), ("HEY", 0.18), ("LAPTOP", 0.14), ("SUP", 0.10),
     ("GREETING", 0.06), ("GOOD-MORNING", 0.04), ("HEY-THERE", 0.03), ("LOUD", 0.01), ("LOUDLY", 0.01)],

    [("MY", 0.28), ("YOUR", 0.23), ("OUR", 0.15), ("THEIR", 0.12), ("WHO", 0.08),
     ("HER", 0.06), ("ITS", 0.04), ("IT", 0.02), ("THEM", 0.01), ("EVERYONE", 0.01)],

    [("LAPTOP", 0.30), ("NAME", 0.25), ("LABEL", 0.15), ("FAME", 0.10), ("CALL", 0.08),
     ("SIGN", 0.05), ("WORD", 0.03), ("NOISE", 0.02), ("THING", 0.01), ("LIGHT", 0.01)],

    [("IS", 0.30), ("BE", 0.25), ("ARE", 0.15), ("EXISTS", 0.12), ("FEELS", 0.08),
     ("APPEARS", 0.05), ("SOUNDS", 0.03), ("RUNS", 0.01), ("JUMPS", 0.00), ("STANDS", 0.01)],

    [("JOHN", 0.32), ("JON", 0.25), ("JACK", 0.15), ("DOG", 0.12), ("CAT", 0.08),
     ("JAMES", 0.04), ("JASON", 0.02), ("JIM", 0.01), ("TREE", 0.00), ("CAR", 0.01)],

    [("AND", 0.28), ("ALSO", 0.23), ("PLUS", 0.15), ("TOO", 0.10), ("LOUD", 0.08),
     ("LOUDLY", 0.05), ("WITH", 0.04), ("BUT", 0.03), ("OR", 0.02), ("YET", 0.02)],

    [("AAAA", 0.26), ("ME", 0.20), ("I", 0.17), ("DOG", 0.12), ("CAT", 0.08),
     ("SELF", 0.05), ("WHO", 0.04), ("THING", 0.03), ("IT", 0.02), ("NOISE", 0.01)],

    [("LOUD", 0.28), ("TRULY", 0.22), ("VERY", 0.18), ("REALLY", 0.12), ("FAR", 0.08),
     ("DEEPLY", 0.05), ("SINCERELY", 0.03), ("NOISY", 0.02), ("BRIGHT", 0.01), ("SHINY", 0.01)],

    [("LIKE", 0.29), ("LOVE", 0.23), ("LOUD", 0.17), ("ENJOY", 0.12), ("NOISE", 0.07),
     ("PREFER", 0.04), ("FAVOR", 0.03), ("APPRECIATE", 0.02), ("HATE", 0.02), ("IGNORE", 0.01)],

    [("RICE", 0.30), ("BURGER", 0.22), ("PIZZA", 0.18), ("SUSHI", 0.12), ("SOUND", 0.07),
     ("PASTA", 0.04), ("SANDWICH", 0.03), ("TREE", 0.02), ("TABLE", 0.01), ("NOISE", 0.01)],
]


candidate_lists = [
    [(word.lower(), score) for word, score in window]
    for window in candidate_lists
]

# You can tweak the parameters below as needed for testing

decoder = TransformerDecoder(
    lm_name="microsoft/phi-4-mini-instruct",
    beam_size=10,
    alpha=0.95,
    beta=0.05,
    device="cuda",
    load_in_8bit=False,
    refine=True,
    debug=True,
)

modes = ["default", "length_norm", "coverage", "uncertainty"]

for mode in modes:
    decoder.scoring_mode = mode
    sentence = decoder.decode(candidate_lists)
    print(f"Mode: {mode}\n→ {sentence}\n")