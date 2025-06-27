"""
Quick functional test for src/Transformer.py

Expected output (approx.):
    >>> Hello my name is john.
"""

from Transformer import TransformerDecoder

# ---------------------------------------------------------------------
# 1) Instantiate the decoder with a tiny model so the test is lightweight
decoder = TransformerDecoder(
        lm_name="distilgpt2",   # 124 M params; loads almost instantly
        beam_size=10,
        alpha=0.6,              # LM weight
        beta=0.4,               # SignCLIP weight
        device="cuda",           # keep test fully CPU-only
)

# ---------------------------------------------------------------------
# 2) Fake SignCLIP top-10 output for five overlapping windows
#    Each inner list is already soft-maxed and sorted (probabilities sum to 1).

candidate_lists = [
    # window-0  → “Hello …”
    [("hello", 0.30), ("hi", 0.18), ("hey", 0.12), ("greetings", 0.10),
     ("hola", 0.08), ("hullo", 0.06), ("salut", 0.05), ("ciao", 0.04),
     ("bonjour", 0.04), ("yo", 0.03)],

    # window-1  → “… my …”
    [("my", 0.35), ("our", 0.20), ("the", 0.12), ("your", 0.09),
     ("ma", 0.07), ("mi", 0.05), ("mein", 0.04), ("mes", 0.03),
     ("mon", 0.03), ("mine", 0.02)],

    # window-2  → “… name …”
    [("name", 0.38), ("title", 0.18), ("label", 0.14), ("nom", 0.08),
     ("navn", 0.06), ("nombre", 0.05), ("naam", 0.04), ("designation", 0.03),
     ("denomination", 0.02), ("appelation", 0.02)],

    # window-3  → “… is …”
    [("is", 0.40), ("be", 0.18), ("equals", 0.14), ("are", 0.10),
     ("was", 0.06), ("’s", 0.04), ("been", 0.03), ("exists", 0.02),
     ("lies", 0.02), ("remains", 0.01)],

    # window-4  → “… John …”
    [("john", 0.42), ("jon", 0.17), ("joe", 0.15), ("johan", 0.08),
     ("joan", 0.05), ("jan", 0.04), ("juan", 0.03), ("sean", 0.03),
     ("ian", 0.02), ("johnny", 0.01)],

    # window-5  → “… and …”
    [("and", 0.34), ("plus", 0.20), ("as well as", 0.12), ("also", 0.10),
     ("ampersand", 0.07), ("n", 0.05), ("along", 0.04), ("with", 0.04),
     ("together", 0.03), ("added", 0.01)],

    # window-6  → “… I …”
    [("i", 0.36), ("me", 0.22), ("myself", 0.15), ("eye", 0.07),
     ("aye", 0.06), ("yo", 0.04), ("ich", 0.03), ("ego", 0.03),
     ("je", 0.02), ("ya", 0.02)],

    # window-7  → “… really …”
    [("really", 0.32), ("truly", 0.19), ("very", 0.16), ("indeed", 0.11),
     ("surely", 0.07), ("certainly", 0.05), ("absolutely", 0.04),
     ("definitely", 0.03), ("positively", 0.02), ("undoubtedly", 0.01)],

    # window-8  → “… like …”
    [("like", 0.37), ("love", 0.18), ("enjoy", 0.14), ("fancy", 0.10),
     ("appreciate", 0.06), ("prefer", 0.05), ("dig", 0.04), ("adore", 0.03),
     ("cherish", 0.02), ("relish", 0.01)],

    # window-9  → “… pizza …”
    [("pizza", 0.39), ("pasta", 0.16), ("burger", 0.14), ("tacos", 0.10),
     ("sushi", 0.07), ("sandwich", 0.05), ("pie", 0.04), ("flatbread", 0.03),
     ("calzone", 0.02), ("lasagna", 0.01)],
]

# ---------------------------------------------------------------------
# 3) Decode and print the sentence
sentence = decoder.decode(candidate_lists)
print(sentence)
