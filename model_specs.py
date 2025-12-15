"""Specifications jackosullivan (jos) convolutional ECG architecture."""

specs = {}

two_d_specs = {
    "jos": ([("C", 64, (1, 15), (0, 7)), ("attention", "channel"), ("C", 128, (1, 15), (0, 7)),
             ("attention", "spatial"), ("m", (1, 4)), ("C", 256, (1, 11), (0, 5)), ("attention", "channel"),
             ("C", 512, (1, 11), (0, 5)), ("attention", "spatial"), ("m", (1, 4)), ("C", 1024, (1, 7), (0, 3)),
             ("attention", "channel"), ("C", 1024, (1, 7), (0, 3)), ("attention", "spatial"), ("m", (1, 2)),
             ("l", 1024), ("a", (1, 1))],
            [1024, "b", "r", "d", 1024, "b", "r", "d"]),
}
