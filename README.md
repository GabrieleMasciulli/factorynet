# FactoryCLIP

## 1. Synthetic Dataset Generation

The simulator generates physics-based motor telemetry data:

```
Current = (Inertia × Acceleration) + (Friction × Velocity) + Noise
```

- **Healthy tools**: Base friction coefficient (0.5)
- **Dull tools**: Elevated friction coefficient (2.5)
- **Training set**: 500 samples of healthy tool operations only
- **Test set**: 200 samples (100 healthy, 100 dull)

The velocity command signal follows sinusoidal patterns mimicking industrial cycles. Motor current is derived from physical principles, creating realistic signal relationships that degrade when tools wear.

## 2. Model Architecture

**FactoryCLIP** uses a dual-encoder architecture:

- **Command Encoder**: Maps velocity commands to a shared latent space
- **Current Encoder**: Maps motor currents to the same latent space

Both encoders are 1D-CNN with batch normalization + projection head in the shared embedding space.

- **Output**: 64-dimensional embeddings

The model learns to align command-current pairs from healthy operations, enabling zero-shot detection when this alignment breaks down due to tool wear.

## 3. Contrastive Loss

InfoNCE loss (CLIP-style):

$$
\text{Logits} = e^{\tau} \cdot \left(Z_{\text{cmd}} Z_{\text{cur}}^{\top}\right), \qquad
\mathcal{L} = \frac{\operatorname{CE}(\text{Logits}, I) + \operatorname{CE}\left(\text{Logits}^{\top}, I\right)}{2}
$$

Where:

- $\tau$ is the learnable logit scale parameter
- Positive pairs are matched by batch index: `(cmd[i], cur[i])`
- Negatives are all other pairs in the batch

This encourages embeddings of aligned command-current pairs to be similar while pushing apart mismatched pairs.

## Usage

```bash
# Generate synthetic data
python src/simulator.py

# Train on healthy data only
python src/train.py

# Evaluate with zero-shot detection
python src/eval.py
```

## Key Insight

By training only on healthy operations, the model learns the expected relationship between commands and currents. Anomalies (dull tools) manifest as reduced cosine similarity in the latent space, enabling detection without anomaly labels during training.
# factorynet
