from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_patchtst import (
    LocalCrossChannelFusion,
    PatchTST,
    TrainConfig,
    masked_bce_loss,
    masked_mse_loss,
)


class PatchTSTSmokeTest(unittest.TestCase):
    def make_base_config(self, model_variant: str) -> TrainConfig:
        return TrainConfig(
            model_variant=model_variant,
            channels=("II", "PLETH", "ABP", "RESP"),
            n_channels=4,
            seq_len=75_000,
            patch_len=64,
            stride=64,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_ff=256,
            cross_channel_layers=1,
            cross_channel_heads=4,
            cross_channel_window=1,
            pooling_type="attention" if model_variant == "patchtst_v2" else "mean",
        )

    def make_v15_config(self, task: str, seq_len: int = 1000) -> TrainConfig:
        return TrainConfig(
            task=task,
            model_variant="patchtst_v1_5",
            channels=("ABP", "II", "PLETH"),
            n_channels=3,
            seq_len=seq_len,
            patch_len=100,
            stride=100,
            d_model=64,
            n_heads=8,
            n_layers=2,
            d_ff=128,
            dropout=0.1,
            attn_dropout=0.0,
            qkv_bias=True,
            pool_depth=1,
            pool_mlp_ratio=4.0,
            pool_num_queries=1,
            pool_complete_block=True,
            pool_affine=False,
        )

    def test_v1_forward_shapes(self):
        config = self.make_base_config("patchtst_v1")
        model = PatchTST(config)
        x = torch.randn(2, 4, 75_000)

        latent, debug = model.forward_features(x, return_debug=True)
        output = model(x)

        self.assertEqual(debug["input"], (2, 4, 75_000))
        self.assertEqual(debug["channel_tokens"], (2, 4, 1171, 128))
        self.assertEqual(debug["encoder_input"], (8, 1171, 128))
        self.assertEqual(debug["latent"], (2, 128))
        self.assertIsNone(debug["local_tokens_per_t"])
        self.assertEqual(tuple(latent.shape), (2, 128))
        self.assertEqual(tuple(output.shape), (2, 1))

    def test_v2_forward_shapes_and_local_fusion(self):
        config = self.make_base_config("patchtst_v2")
        model = PatchTST(config)
        x = torch.randn(2, 4, 75_000)

        latent, debug = model.forward_features(x, return_debug=True)
        output = model(x)

        self.assertEqual(debug["input"], (2, 4, 75_000))
        self.assertEqual(debug["channel_tokens"], (2, 4, 1171, 128))
        self.assertEqual(debug["encoder_input"], (8, 1171, 128))
        self.assertEqual(debug["fused_tokens"], (2, 1171, 128))
        self.assertEqual(debug["latent"], (2, 128))
        self.assertEqual(debug["local_tokens_per_t"], 12)
        self.assertEqual(tuple(latent.shape), (2, 128))
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertEqual(model.fusion_layers[0].offset_embed.num_embeddings, 3)

    def test_no_regression_construct_and_forward_v1_v2(self):
        x = torch.randn(2, 4, 75_000)
        for variant in ["patchtst_v1", "patchtst_v2"]:
            model = PatchTST(self.make_base_config(variant))
            y = model(x)
            self.assertEqual(tuple(y.shape), (2, 1))

    def test_local_cross_channel_fusion_shape_and_offset_embeddings(self):
        fusion = LocalCrossChannelFusion(
            d_model=128,
            n_heads=4,
            d_ff=256,
            dropout=0.1,
            window=1,
        )
        x = torch.randn(2, 4, 1171, 128)

        local_tokens = fusion._local_tokens(x)
        fused = fusion(x)

        self.assertEqual(fusion.offset_embed.num_embeddings, 3)
        self.assertEqual(tuple(local_tokens.shape), (2 * 1171, 12, 128))
        self.assertEqual(tuple(fused.shape), (2, 1171, 128))

    def test_v2_rejects_multiple_cross_channel_layers(self):
        bad_config = self.make_base_config("patchtst_v2")
        bad_config.cross_channel_layers = 2
        with self.assertRaisesRegex(ValueError, "cross_channel_layers=1"):
            PatchTST(bad_config)

    def test_v15_forward_shapes(self):
        config = self.make_v15_config(task="event", seq_len=1000)
        model = PatchTST(config)
        x = torch.randn(2, 3, 1000)

        latent, debug = model.forward_features(x, return_debug=True)
        output = model(x)

        self.assertEqual(debug["input"], (2, 3, 1000))
        self.assertEqual(debug["patch_tokens"], (2, 3, 10, 64))
        self.assertEqual(debug["encoder_input"], (6, 10, 64))
        self.assertEqual(debug["encoder_output"], (2, 3, 10, 64))
        self.assertEqual(debug["pooler_raw_output"], (6, 1, 64))
        self.assertEqual(debug["pooled_channels"], (2, 3, 64))
        self.assertEqual(debug["latent"], (2, 192))
        self.assertEqual(tuple(latent.shape), (2, 192))
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertEqual(model.v15_patch_embed.proj.groups, 3)
        self.assertEqual(model.n_patches, 10)

    def test_v15_padding_patch_count_matches_tokenizer_output(self):
        config = self.make_v15_config(task="event", seq_len=1050)
        model = PatchTST(config)
        x = torch.randn(2, 3, 1050)

        tokens = model.v15_patch_embed(x)

        self.assertEqual(model.v15_patch_embed.num_patches(1050), 11)
        self.assertEqual(model.n_patches, 11)
        self.assertEqual(tuple(tokens.shape), (2, 3, 11, 64))

    def test_v15_gradient_flow_sum_loss(self):
        torch.manual_seed(0)
        model = PatchTST(self.make_v15_config(task="event", seq_len=1000))
        model.train()
        pred = model(torch.randn(2, 3, 1000))
        loss = pred.float().sum()
        loss.backward()

        params = {
            "tokenizer": model.v15_patch_embed.proj.weight,
            "W_Q": model.v15_encoder[0].self_attn.W_Q.weight,
            "W_K": model.v15_encoder[0].self_attn.W_K.weight,
            "W_V": model.v15_encoder[0].self_attn.W_V.weight,
            "ff0": model.v15_encoder[0].ff[0].weight,
            "pool_query": model.v15_pooler.query_tokens,
            "pool_W_Q": model.v15_pooler.cross_attention_block.xattn.W_Q.weight,
            "head": model.v15_head.weight,
        }
        for name, param in params.items():
            self.assertIsNotNone(param.grad, f"missing gradient for {name}")
            self.assertTrue(torch.isfinite(param.grad).all(), f"non-finite gradient for {name}")

    def test_v15_event_loss_and_backward(self):
        torch.manual_seed(0)
        model = PatchTST(self.make_v15_config(task="event", seq_len=1000))
        x = torch.randn(2, 3, 1000)
        pred = model(x)
        target = torch.tensor([1.0, 0.0])
        mask = torch.tensor([True, True])
        loss = masked_bce_loss(pred, target, mask)
        model.zero_grad(set_to_none=True)
        loss.backward()

        self.assertEqual(tuple(pred.shape), (2, 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.v15_head.weight.grad)
        self.assertTrue(torch.isfinite(model.v15_head.weight.grad).all())

    def test_v15_feature_loss_and_backward(self):
        torch.manual_seed(0)
        model = PatchTST(self.make_v15_config(task="feature", seq_len=1000))
        x = torch.randn(2, 3, 1000)
        pred = model(x)
        target = torch.tensor([0.5, -0.25])
        mask = torch.tensor([True, True])
        loss = masked_mse_loss(pred, target, mask)
        model.zero_grad(set_to_none=True)
        loss.backward()

        self.assertEqual(tuple(pred.shape), (2, 1))
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.v15_head.weight.grad)
        self.assertTrue(torch.isfinite(model.v15_head.weight.grad).all())

    def test_v15_rejects_unsupported_pool_and_bad_dims(self):
        with self.assertRaisesRegex(ValueError, "pool_depth=1"):
            PatchTST(replace(self.make_v15_config(task="event", seq_len=1000), pool_depth=2))

        with self.assertRaisesRegex(ValueError, "pool_num_queries=1"):
            PatchTST(replace(self.make_v15_config(task="event", seq_len=1000), pool_num_queries=2))

        with self.assertRaisesRegex(ValueError, "pool_affine=False"):
            PatchTST(replace(self.make_v15_config(task="event", seq_len=1000), pool_affine=True))

        with self.assertRaisesRegex(ValueError, "d_model divisible by n_heads"):
            PatchTST(replace(self.make_v15_config(task="event", seq_len=1000), d_model=62, n_heads=8))

        with self.assertRaisesRegex(ValueError, "patch_len > 0"):
            PatchTST(replace(self.make_v15_config(task="event", seq_len=1000), patch_len=0))

        with self.assertRaisesRegex(ValueError, "stride > 0"):
            PatchTST(replace(self.make_v15_config(task="event", seq_len=1000), stride=0))


if __name__ == "__main__":
    unittest.main()
