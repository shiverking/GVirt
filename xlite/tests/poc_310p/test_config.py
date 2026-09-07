import json
import unittest

from xlite.poc_310p import extract_text_config, load_qwen3_asr_llm_args


TEXT_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "rope_parameters": {
        "rope_type": "mrope",
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": True,
    },
}


class TestQwen3AsrConfig(unittest.TestCase):
    def test_extracts_nested_asr_text_config(self):
        actual = extract_text_config({"thinker_config": {"text_config": TEXT_CONFIG}})
        self.assertIs(actual, TEXT_CONFIG)

    def test_maps_asr_config_to_single_card_fp16_args(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir)
            (checkpoint / "config.json").write_text(
                json.dumps({"thinker_config": {"text_config": TEXT_CONFIG}}), encoding="utf-8"
            )
            args = load_qwen3_asr_llm_args(checkpoint, max_seq_len=256)
        self.assertEqual(args["dtype"], "float16")
        self.assertEqual(args["max_batch_size"], 1)
        self.assertEqual(args["max_seq_len"], 256)
        self.assertEqual(args["dim"], 2048)
        self.assertEqual(args["head_dim"], 128)
        self.assertTrue(args["qk_norm"])
        self.assertEqual(args["rope_type"], "mrope")
        self.assertEqual(args["mrope_section"], [24, 20, 20])
        self.assertTrue(args["mrope_interleaved"])

    def test_rejects_incomplete_text_config(self):
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            extract_text_config({"text_config": {"hidden_size": 2048}})

    def test_rejects_invalid_mrope_partition(self):
        import tempfile
        from pathlib import Path

        invalid = dict(TEXT_CONFIG)
        invalid["rope_parameters"] = {"mrope_section": [1, 2, 3]}
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir)
            (checkpoint / "config.json").write_text(
                json.dumps({"thinker_config": {"text_config": invalid}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "head_dim / 2"):
                load_qwen3_asr_llm_args(checkpoint)


if __name__ == "__main__":
    unittest.main()
