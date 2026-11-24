# Mnemosyne VAE

**DeathAI Australia** — built by Daniel Newton

Mnemosyne (Greek goddess of memory) is a Variational Autoencoder system built for the DeathAI stack. Encodes inputs into a latent probability distribution, preserving the essence of data while allowing controlled reconstruction and generation.

Named deliberately — a VAE doesn't memorise, it learns the shape of memory. Same idea.

## What it does

- Convolutional VAE architecture (PyTorch)
- Async training and generation support
- Encode → latent space → decode/generate pipeline
- Built for 28x28 RGB images, designed to be retargeted
- Part of the ZeroTraceGPT local image toolchain

## Files

- `37_MNEMOSYNE_DeathAIAUS.py` — full implementation
- `train_mnemosyne.py` — training script

## Stack

Python 3.10+, PyTorch, NumPy, Matplotlib

## Context

Part of [DeathAI Australia](https://deathaiaustralia.com.au) — local-first AI infrastructure. No cloud dependency. Runs on consumer hardware.

Also see: [Baron Samedi K-means](https://github.com/deathdopest-sketch/baron-samedi-kmeans) — the clustering model built alongside this.
