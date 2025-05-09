# 🕵️‍♂️ Bot Detection Architecture

This repository presents two advanced bot detection architectures for analyzing user behavior and interaction patterns on social platforms. The approaches combine content, temporal, behavioral, and graph-based features to detect inauthentic activity using both semi-supervised and self-supervised learning methods.

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture 1: Multimodal Semi-Supervised](#architecture-1-multimodal-semi-supervised)
- [Architecture 2: GMAE2-CGNN (Self-Supervised)](#architecture-2-gmae2-cgnn-self-supervised)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Visualizations](#visualizations)
- [References](#references)

---

## 📖 Overview

The goal of this project is to detect bots by leveraging:
- User behavior patterns
- Posting regularity
- Engagement metrics
- Network structure

It supports both:
- **Post-level detection** (individual post suspiciousness)
- **Author-level aggregation** (final bot label for users)

---

## 🧠 Architecture 1: Multimodal Semi-Supervised

1. **User Filtering** – Only users with ≥2 posts.
2. **Feature Extraction** – Extracted from content, temporal, behavioral, and graph features.
3. **Content Similarity Detection** – TF-IDF + cosine similarity.
4. **Pseudo-Labeling** – Isolation Forest, DBSCAN, KMeans, PCA-Mahalanobis.
5. **Supervised Learning** – ExtraTreesClassifier trained on pseudo-labels.
6. **Final Classification** – Based on model prediction and optimal thresholding.
7. **Aggregation** – Post-level results summarized at user level.
8. **Visualization & Reporting** – Includes heatmaps, histograms, feature importance, etc.

---

## 🧬 Architecture 2: GMAE2-CGNN (Self-Supervised)

1. **Preprocessing** – Filter users with fewer than 2 posts.
2. **Feature Extraction** – Standardized features include text entropy, sentiment, posting time, etc.
3. **Graph Construction** – Nodes = users, Edges = mentions or high similarity posts.
4. **Model Design**
   - Feature Encoder: MLP (128→64→32)
   - Graph Encoder: GNN (32→16)
5. **Training** – Contrastive loss on graph structure (no labels).
6. **Bot Detection** – Clustering (KMeans) on learned embeddings.
7. **Output** – Embeddings, cluster labels, bot likelihood.

---

## ✅ Key Features

- Posting frequency and burst detection
- Time entropy and regularity scoring
- Benford’s Law deviation for engagement
- Graph-based centrality and community metrics
- Contrastive self-supervised learning with GNN

