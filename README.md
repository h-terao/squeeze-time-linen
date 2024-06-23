# SqueezeTime

Unofficial Flax implementation of SqueezeTime model.  
Official implementation can be found [here](https://github.com/xinghaochen/SqueezeTime/tree/main).

## Prerequests
- jax
- jaxlib
- flax
- chex
- einops

I used Poetry, but you must update dependencies to use JAX on GPUs/TPUs.

## Pretrained variables

I converted [official pretrained variables](https://github.com/xinghaochen/SqueezeTime/tree/main) into this implementation using `notebooks/convert.ipynb`.  
You can also download the converted variables from [here](https://drive.google.com/drive/folders/17KX5RKGecMEWf5_NrA0VlhrR5_d7JH2x?usp=sharing).

After downloading the converted variables, you can load them as follows:
```python
from pathlib import Path
import pickle

ckpt_path = Path("k400.pkl")
variables = pickle.loads(ckpt_path.read_bytes())
```

## Inference

After loading the converted variables, you can use SqueezeTime for inference.

```python
import jax.numpy as jnp
from squeeze_time_linen import resnet50

# Instantiate model
flax_model = resnet50(num_classes=400)

# Video array of shape (*batch_dims, time, height, width, channels)
# Time should be 16 to use pretrained variables.
x = jnp.zeros((16, 224, 224, 3))

# Normalize
mean = jnp.array([123.675, 116.28, 103.53])
std = jnp.array([58.395, 57.12, 57.375])
x = (x - mean) / std

# Forward
logits = flax_model.apply(variables, x, is_training=False)
```

## Citation

```bibtex
@article{zhai2024SqueezeTime,
  title={No Time to Waste: Squeeze Time into Channel for Mobile Video Understanding},
  author={Zhai, Yingjie and Li, Wenshuo and Tang, Yehui and Chen, Xinghao and Wang, Yunhe},
  journal={arXiv preprint arXiv:2405.08344},
  year={2024}
}
```
