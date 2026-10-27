# Copyright 2025, Lawrence Livermore National Security, LLC and professor
# contributors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch
import torch.autograd.forward_ad as fwAD
import torch.nn as nn
from omegaconf import OmegaConf

from professor.__main__ import GENERATOR_ACTIVATION_FUNCTIONS
from professor.layers import AddCoords, AlphaLinear, ResBlock
from professor.torch_models import GenResizeResNet, build_generator
from professor.vela.config import FullSchema, Keys, build_template_config
from professor.vela.utils import get_object_reference, initialize_function_calls_in_parameters

torch.manual_seed(1231231)


class TestGenResizeResNet(unittest.TestCase):

    def test_output_shape(self):
        X = torch.rand(2, 4, 1, 1)
        for y_kernel, x_kernel, im_size in [(4, 4, 64), (2, 1, 64), (6, 2, 768), (2, 6, 96)]:
            model = GenResizeResNet(
                4, im_size, 3, min_features=8, max_features=32, y_kernel=y_kernel, x_kernel=x_kernel
            )
            max_kernel = max(y_kernel, x_kernel)
            shape = (2, 3, im_size * y_kernel // max_kernel, im_size * x_kernel // max_kernel)
            self.assertEqual(tuple(model(X).shape), shape)

    def test_blocks_per_stage(self):
        model = GenResizeResNet(4, 64, 1, min_features=8, max_features=32, blocks=[0, 0, 0, 1])
        self.assertEqual(sum(isinstance(m, ResBlock) for m in model.modules()), 1)
        self.assertEqual(tuple(model(torch.rand(2, 4, 1, 1)).shape), (2, 1, 64, 64))
        with self.assertRaises(ValueError):
            GenResizeResNet(4, 64, 1, blocks=[1, 1])

    def test_im_size_must_match_kernel(self):
        with self.assertRaises(ValueError):
            GenResizeResNet(4, 96, 1, y_kernel=4, x_kernel=4)

    def test_kernels_must_match_field_aspect_ratio(self):
        # 3 x 2 kernels reach im_size 768 on the long side, but give 768 x 512, not the fields' 768 x 256
        with self.assertRaisesRegex(ValueError, "768 x 512 output, but the fields are 768 x 256"):
            GenResizeResNet(4, 768, 1, y_kernel=3, x_kernel=2, y_pixels=768, x_pixels=256)
        model = GenResizeResNet(
            4, 768, 1, min_features=4, max_features=8, y_kernel=3, x_kernel=1, y_pixels=768, x_pixels=256
        )
        self.assertEqual(tuple(model(torch.rand(2, 4, 1, 1)).shape), (2, 1, 768, 256))
        with self.assertRaises(ValueError):
            build_generator(
                "2D-resize-resnet", input_size=4, im_size=768, y_kernel=1, x_kernel=3, y_pixels=768, x_pixels=256
            )

    def test_res_block_starts_as_identity(self):
        for use_batch_norm in (False, True):
            block = ResBlock(8, nn.SiLU(inplace=True), use_batch_norm).eval()
            x = torch.randn(2, 8, 5, 5)
            self.assertTrue(torch.equal(block(x), x))

    def test_coords(self):
        out = AddCoords()(torch.zeros(3, 1, 4, 2))
        self.assertEqual(tuple(out.shape), (3, 3, 4, 2))
        self.assertTrue(torch.equal(out[0, 1, :, 0], torch.linspace(-1, 1, 4)))
        self.assertTrue(torch.equal(out[0, 2, 0], torch.tensor([-1.0, 1.0])))

    def test_no_coords_no_batch_norm(self):
        model = GenResizeResNet(4, 16, 1, min_features=4, max_features=8, use_batch_norm=False, coord_channels=False)
        self.assertFalse(any(isinstance(m, (AddCoords, nn.BatchNorm2d)) for m in model.modules()))
        self.assertEqual(tuple(model(torch.rand(2, 4, 1, 1)).shape), (2, 1, 16, 16))

    def test_activation_functions(self):
        X = torch.rand(2, 4, 1, 1)
        for act_fun in GENERATOR_ACTIVATION_FUNCTIONS:
            GenResizeResNet(4, 16, 1, min_features=4, max_features=8, act_fun=act_fun)(X)

    def test_forward_mode_jvp(self):
        """forward-mode derivatives d output / d input match finite differences"""
        model = GenResizeResNet(
            3, 32, 2, min_features=4, max_features=8, use_batch_norm=False, last_layer=nn.Identity(), act_fun="SiLU"
        ).double()
        for p in model.parameters():  # move the zero-initialized blocks off the identity
            p.data.add_(0.05 * torch.randn_like(p))
        x, v = torch.rand(2, 3, 1, 1, dtype=torch.float64), torch.randn(2, 3, 1, 1, dtype=torch.float64)
        with fwAD.dual_level():
            jvp = fwAD.unpack_dual(model(fwAD.make_dual(x, v))).tangent
        eps = 1e-6
        fd = (model(x + eps * v) - model(x - eps * v)) / (2 * eps)
        self.assertTrue(torch.allclose(jvp, fd, rtol=1e-5, atol=1e-8))

    def test_build_generator(self):
        model = build_generator(
            "2D-resize-resnet",
            input_size=4,
            im_size=48,
            num_channels=3,
            min_features=4,
            max_features=16,
            last_layer=AlphaLinear(n_channels=3),
            use_batch_norm=False,
            y_kernel=3,
            x_kernel=1,
            act_fun="SiLU",
            res_blocks=[0, 0, 1, 2],
        )
        self.assertIsInstance(model, GenResizeResNet)
        self.assertEqual(sum(isinstance(m, ResBlock) for m in model.modules()), 3)
        self.assertEqual(tuple(model(torch.rand(2, 4, 1, 1)).shape), (2, 3, 48, 16))


def test_vela_template_config_builds_the_model(tmp_path):
    """a --vis-config for the generator rebuilds the same model, including for non-square fields"""
    config_path = tmp_path / "template_config.yaml"
    build_template_config(
        config_fname=str(config_path),
        n_inputs=4,
        n_channels=3,
        y_pixels=48,
        x_pixels=16,
        min_features=4,
        max_features=16,
        y_kernel=3,
        x_kernel=1,
        generator_type="2D-resize-resnet",
        act_fun="SiLU",
        use_batch_norm=False,
        res_blocks=[0, 0, 1, 2],
    )
    conf = FullSchema(OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)).val
    invocation = conf[Keys.MODEL.value][Keys.PYTORCH.value][Keys.INVOCATION.value]
    model = get_object_reference(invocation[Keys.TARGET.value])(
        **initialize_function_calls_in_parameters(invocation[Keys.PARAMS.value])
    )
    reference = build_generator(
        "2D-resize-resnet",
        input_size=4,
        im_size=48,
        num_channels=3,
        min_features=4,
        max_features=16,
        last_layer=AlphaLinear(n_channels=3),
        use_batch_norm=False,
        y_kernel=3,
        x_kernel=1,
        act_fun="SiLU",
        res_blocks=[0, 0, 1, 2],
    )
    model.load_state_dict(reference.state_dict())
    assert tuple(model(torch.rand(2, 4, 1, 1)).shape) == (2, 3, 48, 16)


if __name__ == "__main__":
    unittest.main()
