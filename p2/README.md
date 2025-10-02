# Chimney Height Regressor

DINOv3-based regressor for predicting chimney height from satellite imagery + metadata. Frozen backbone with trainable MLP head.
Supports 3 fusion types: `baseline` (CLS concat), `cross_attn` (attention over all tokens), `film` (feature modulation, memory efficient).
