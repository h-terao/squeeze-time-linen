from typing import Any, Sequence
from functools import partial

import jax
import jax.numpy as jnp
from flax import linen
import chex
import einops

ModuleDef = Any


def to_2tuple(x: Any | Sequence[Any]) -> tuple[Any, Any]:
    if isinstance(x, Sequence):
        assert len(x) == 2
        return x
    return (x, x)


def to_padding(padding: int | tuple[int, int] | Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if isinstance(padding, int):
        return [(padding, padding), (padding, padding)]
    elif isinstance(padding, tuple) and isinstance(padding[0], int):
        return [padding, padding]
    else:
        assert isinstance(padding, Sequence) and len(padding) == 2
        for x in padding:
            assert isinstance(x, Sequence) and len(x) == 2
        return [tuple[p] for p in padding]


def resize_with_aligned_corners(
    image: chex.Array,
    shape: Sequence[int],
    method: str | jax.image.ResizeMethod,
    antialias: bool,
):
    """Alternative to jax.image.resize(), which emulates align_corners=True in PyTorch's
    interpolation functions.

    Copy from https://github.com/google/jax/issues/11206#issuecomment-1423140760
    """
    spatial_dims = tuple(i for i in range(len(shape)) if not jax.core.symbolic_equal_dim(image.shape[i], shape[i]))
    scale = jnp.array([(shape[i] - 1.0) / (image.shape[i] - 1.0) for i in spatial_dims])
    translation = -(scale / 2.0 - 0.5)
    return jax.image.scale_and_translate(
        image,
        shape,
        method=method,
        scale=scale,
        spatial_dims=spatial_dims,
        translation=translation,
        antialias=antialias,
    )


class GlobalConv(linen.Module):
    """Top branch in IOI module."""

    features: int
    num_frames: int = 16
    pos_dim: int = 16
    conv: ModuleDef = linen.Conv
    norm: ModuleDef = linen.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array, param: chex.Array) -> chex.Array:
        # Temporal focus convolution.
        x = x * param
        x = self.conv(self.num_frames, kernel_size=(3, 3), padding=to_padding(1), name="conv1")(x)
        x = self.norm(name="norm1")(x)
        x = linen.relu(x)

        # Time encoding
        *_, h, w, _ = jnp.shape(x)
        x += resize_with_aligned_corners(
            self.param("pos_embed", linen.initializers.kaiming_normal(), (self.pos_dim, self.pos_dim, self.num_frames)),
            shape=(h, w, self.num_frames),
            method="bilinear",
            antialias=False,
        )

        x = self.conv(self.num_frames, kernel_size=(7, 7), padding=to_padding(3), name="conv2")(x)
        x = self.norm(name="norm2")(x)
        x = linen.relu(x)

        x = self.conv(self.features, kernel_size=(3, 3), padding=to_padding(1), name="conv3")(x)
        x = linen.sigmoid(x)
        return x


class IOI(linen.Module):
    """
    Inter-temporal object interaction module.
    """

    features: int
    num_frames: int = 16
    pos_dim: int = 16
    conv: ModuleDef = linen.Conv
    norm: ModuleDef = linen.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array, param: chex.Array) -> chex.Array:
        # Top branch of IOI module.
        x_glo = GlobalConv(
            self.features,
            num_frames=self.num_frames,
            pos_dim=self.pos_dim,
            conv=self.conv,
            norm=self.norm,
            name="glo_conv",
        )(x, param)

        # Bottom branch of IOI module.
        x_short = self.conv(
            self.features,
            kernel_size=(3, 3),
            padding=to_padding(1),
            name="short_conv",
        )(x)

        return x_short * x_glo


class ParamConv(linen.Module):
    """A module that calculates the temporal-adaptive weights."""

    conv: ModuleDef = linen.Conv
    norm: ModuleDef = linen.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        in_features = jnp.size(x, axis=-1)
        param = einops.reduce(x, "... h w c -> ... 1 1 c", "mean")
        param = self.conv(in_features, kernel_size=(1, 1), use_bias=False, name="conv1")(param)
        param = self.norm(name="norm1")(param)
        param = linen.relu(param)
        param = self.conv(in_features, kernel_size=(1, 1), use_bias=False, name="conv2")(param)
        param = linen.sigmoid(param)
        return param


class CTL(linen.Module):
    """Channel-Time Learning block."""

    features: int
    num_frames: int = 16
    pos_dim: int = 7
    kernel_size: int | tuple[int, int] = 1
    padding: int = 0
    feature_group_count: int = 1
    use_bias: bool = True
    conv: ModuleDef = linen.Conv
    norm: ModuleDef = linen.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        # Calculate temporal-adaptive weights
        param = ParamConv(conv=self.conv, norm=self.norm, name="param_conv")(x)

        # Temporal focus convolution
        x_temporal = self.conv(
            self.features,
            kernel_size=to_2tuple(self.kernel_size),
            padding=to_padding(self.padding),
            feature_group_count=self.feature_group_count,
            use_bias=self.use_bias,
            name="temporal_conv",
        )(x * param)

        x_spatial = IOI(
            self.features,
            num_frames=self.num_frames,
            pos_dim=self.pos_dim,
            conv=self.conv,
            norm=self.norm,
            name="spatial_conv",
        )(x, param)

        return x_temporal + x_spatial


class BasicBlock(linen.Module):
    features: int
    num_frames: int = 16
    stride: int = 1
    pos_dim: int = 7
    conv: ModuleDef = linen.Conv
    norm: ModuleDef = linen.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if self.stride != 1:
            assert self.stride == 2
            in_features = jnp.size(x, axis=-1)
            x = self.conv(
                in_features,
                kernel_size=(2, 2),
                strides=self.stride,
                feature_group_count=in_features,
                padding="VALID",
                name="downsample.0",
            )(x)
            x = self.norm(name="downsample.1")(x)

        # FIXME: pos_dim is not set here in official impl.
        # is it a bug?
        h = CTL(
            self.features,
            num_frames=self.num_frames,
            # pos_dim=self.pos_dim,
            kernel_size=1,
            padding=0,
            use_bias=False,
            conv=self.conv,
            norm=self.norm,
            name="conv1",
        )(x)
        h = self.norm(name="bn1")(h)
        h = linen.relu(h)

        h = CTL(
            self.features,
            num_frames=self.num_frames,
            pos_dim=self.pos_dim,
            kernel_size=1,
            padding=0,
            use_bias=False,
            conv=self.conv,
            norm=self.norm,
            name="conv2",
        )(h)
        h = self.norm(name="bn2")(h)

        if jnp.shape(x) != jnp.shape(h):
            x = self.conv(self.features, kernel_size=(1, 1), use_bias=False, name="shortcut_conv.0")(x)
            x = self.norm(name="shortcut_conv.1")(x)

        y = linen.relu(x + h)
        return y


class Bottleneck(linen.Module):
    features: int
    num_frames: int = 16
    stride: int = 1
    pos_dim: int = 7
    expansion: int = 4
    conv: ModuleDef = linen.Conv
    norm: ModuleDef = linen.BatchNorm

    @linen.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        if self.stride != 1:
            assert self.stride == 2
            in_features = jnp.size(x, axis=-1)
            x = self.conv(
                in_features,
                kernel_size=(2, 2),
                strides=self.stride,
                feature_group_count=in_features,
                padding="VALID",
                name="downsample.0",
            )(x)
            x = self.norm(name="downsample.1")(x)

        h = self.conv(self.features, kernel_size=(1, 1), use_bias=False, name="conv1")(x)
        h = self.norm(name="bn1")(h)
        h = linen.relu(h)

        h = CTL(
            self.features,
            num_frames=self.num_frames,
            pos_dim=self.pos_dim,
            kernel_size=1,
            padding=0,
            use_bias=False,
            conv=self.conv,
            norm=self.norm,
            name="conv2",
        )(h)
        h = self.norm(name="bn2")(h)
        h = linen.relu(h)

        h = self.conv(self.features * self.expansion, kernel_size=(1, 1), use_bias=False, name="conv3")(h)
        h = self.norm(name="bn3")(h)

        if jnp.shape(x) != jnp.shape(h):
            x = self.conv(self.features * self.expansion, kernel_size=(1, 1), use_bias=False, name="shortcut_conv.0")(x)
            x = self.norm(name="shortcut_conv.1")(x)

        y = linen.relu(x + h)
        return y


class ResNet(linen.Module):
    """SqueezeTime ResNet model.

    Attributes:
        stage_sizes: Number of blocks in each stage.
        block_cls: Residual block class.
        num_classes: Number of classes.
            If 0, the final dense layer is not added.
        num_frames: Number of frames.
        drop_rate: Dropout rate.
        widen_factor: Width factor.
        pos_dims: Positional embedding dimensions.
        dtype: Data type for computation.
        norm_dtype: Data type for normalization.
        param_dtype: Data type for parameters.
    """

    stage_sizes: list[int]
    block_cls: ModuleDef
    num_classes: int = 400
    num_frames: int = 16
    drop_rate: float = 0.5
    widen_factor: float = 1.0
    pos_dims: list[int] = (56, 28, 14, 7)
    dtype: chex.ArrayDType = jnp.float32
    norm_dtype: chex.ArrayDType = jnp.float32
    param_dtype: chex.ArrayDType = jnp.float32

    @linen.compact
    def __call__(self, x: chex.Array, is_training: bool = False) -> chex.Array:
        base_size = int(64 * self.widen_factor)
        conv = partial(linen.Conv, dtype=self.dtype, param_dtype=self.param_dtype)
        norm = partial(
            linen.BatchNorm, use_running_average=not is_training, dtype=self.norm_dtype, param_dtype=self.param_dtype
        )

        x = einops.rearrange(x, "... T H W C -> ... H W (C T)")
        x = conv(base_size, kernel_size=(5, 5), strides=2, padding=to_padding(2), use_bias=False, name="conv1")(x)
        x = norm(name="bn1")(x)
        x = linen.relu(x)
        x = linen.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding=to_padding(1))

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                x = self.block_cls(
                    features=base_size * 2**i,
                    num_frames=self.num_frames,
                    stride=2 if i > 0 and j == 0 else 1,
                    pos_dim=self.pos_dims[i],
                    conv=conv,
                    norm=norm,
                    name=f"layer{i+1}.{j}",
                )(x)

        x = einops.reduce(x, "... H W C -> ... C", "mean")
        x = linen.Dropout(rate=self.drop_rate, deterministic=not is_training)(x)
        if self.num_classes > 0:
            x = linen.Dense(self.num_classes, dtype=self.dtype, param_dtype=self.param_dtype, name="fc")(x)

        return x


def resnet18(**kwargs) -> ResNet:
    return ResNet(stage_sizes=[2, 2, 2, 2], block_cls=BasicBlock, **kwargs)


def resnet34(**kwargs) -> ResNet:
    return ResNet(stage_sizes=[3, 4, 6, 3], block_cls=BasicBlock, **kwargs)


def resnet50(**kwargs) -> ResNet:
    return ResNet(stage_sizes=[3, 4, 6, 3], block_cls=Bottleneck, **kwargs)


def resnet101(**kwargs) -> ResNet:
    return ResNet(stage_sizes=[3, 4, 23, 3], block_cls=Bottleneck, **kwargs)


def resnet152(**kwargs) -> ResNet:
    return ResNet(stage_sizes=[3, 8, 36, 3], block_cls=Bottleneck, **kwargs)


def resnet200(**kwargs) -> ResNet:
    return ResNet(stage_sizes=[3, 24, 36, 3], block_cls=Bottleneck, **kwargs)
