from collections import defaultdict
from re import L

import matplotlib.pyplot as plt
import torch

from visualize.visualizer import Visualizer

class ConceptEmbeddingVisualizer(Visualizer):
    per_batch = False

    def visualize(self, results, model, iteration, **kwargs):
        embeddings = results["embedding"]
        labels = results["label"]

        # dimension_stddev
        if model.rep == "box":
            dim = embeddings.shape[-1] / 2
            anchors = embeddings[:, :dim]