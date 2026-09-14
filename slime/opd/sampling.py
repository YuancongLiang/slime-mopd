"""Checkpointable weighted domain sampling over an existing tokenized dataset."""

import random

from .protocol import digest


class DomainSampler:
    def __init__(self, samples, config, seed, shuffle=True):
        self.config_hash = config["sampling_fingerprint"]
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.domains = list(config["domains"])
        self.weights = [config["domains"][d]["sampling_weight"] for d in self.domains]
        self.indices = {d: [] for d in self.domains}
        for i, sample in enumerate(samples):
            domain = (sample.metadata or {}).get(config["domain_key"])
            if domain not in self.indices:
                raise ValueError(f"Dataset sample {i} has unconfigured MOPD domain {domain!r}")
            self.indices[domain].append(i)
        for d, weight in zip(self.domains, self.weights, strict=True):
            if weight and not self.indices[d]:
                raise ValueError(f"Domain {d} has positive weight but no prompts")
        self.dataset_hash = digest(
            [
                [
                    d,
                    [
                        (i, digest([getattr(samples[i], "prompt", None), getattr(samples[i], "label", None)]))
                        for i in indices
                    ],
                ]
                for d, indices in self.indices.items()
            ]
        )
        self.cursors = dict.fromkeys(self.domains, 0)
        if shuffle:
            for indices in self.indices.values():
                self.rng.shuffle(indices)

    def draw(self, count):
        result = []
        for _ in range(count):
            domain = self.rng.choices(self.domains, weights=self.weights)[0]
            indices = self.indices[domain]
            cursor = self.cursors[domain]
            if cursor == len(indices):
                if self.shuffle:
                    self.rng.shuffle(indices)
                cursor = 0
            result.append(indices[cursor])
            self.cursors[domain] = cursor + 1
        return result

    def state_dict(self):
        return {
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "rng": self.rng.getstate(),
            "indices": {d: list(v) for d, v in self.indices.items()},
            "cursors": dict(self.cursors),
        }

    def load_state_dict(self, state):
        if state["config_hash"] != self.config_hash or state["dataset_hash"] != self.dataset_hash:
            raise ValueError("MOPD sampler config/dataset changed; cannot resume exactly")
        self.rng.setstate(state["rng"])
        self.indices = state["indices"]
        self.cursors = state["cursors"]
