import pytest

from tests.fast.launch_scripts.py_harness import freeze_environment, import_launch_script, install_command_recorder
from tests.fast.launch_scripts.sh_harness import REPO_ROOT

FULL_MODEL = "MiMo-V2.6-Flash-RL-bf16"


@pytest.mark.parametrize(
    "precision, gpus_per_engine, engine_checkpoint",
    [("bf16", 8, None), ("fp8", 4, "MiMo-V2.6-Flash-RL-FP8"), ("fp8-mxfp4", 2, "MiMo-V2.6-Flash-RL")],
)
def test_full_model_engines_follow_the_rollout_precision(
    monkeypatch, tmp_path, precision, gpus_per_engine, engine_checkpoint
):
    freeze_environment(monkeypatch)
    recording = install_command_recorder(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash_tinker.py")
    model_dir = tmp_path / "models"
    cli_args = dict(model_name=FULL_MODEL, rollout_precision=precision, model_dir=str(model_dir))

    launcher.prepare(**cli_args)
    launcher.serve(**cli_args)

    conversions = [cmd for cmd in recording.commands if "convert_mimo_v2_to_bf16.py" in cmd]
    assert f"--save-dir {model_dir}/{FULL_MODEL} " in conversions[0]
    fp8_conversions = [cmd for cmd in conversions if "--fp8-experts" in cmd]
    assert len(fp8_conversions) == (precision == "fp8")
    assert all(f"--save-dir {model_dir}/MiMo-V2.6-Flash-RL-FP8 " in cmd for cmd in fp8_conversions)
    (serve,) = [cmd for cmd in recording.commands if "ray job submit" in cmd]
    assert f"--hf-checkpoint {model_dir}/{FULL_MODEL} " in serve
    assert f"--rollout-num-gpus-per-engine {gpus_per_engine} " in serve
    assert "--sglang-speculative-algorithm EAGLE " in serve
    if engine_checkpoint is None:
        assert "--sglang-config" not in serve and not recording.pseudo_files
    else:
        (engine_config,) = recording.pseudo_files
        assert f"model_path: {model_dir}/{engine_checkpoint}\n" in engine_config


@pytest.mark.parametrize("precision", ["fp8", "fp8-mxfp4"])
def test_partial_model_serves_bf16_only(monkeypatch, precision):
    freeze_environment(monkeypatch)
    launcher = import_launch_script(REPO_ROOT / "scripts/run_mimo_v2_6_flash_tinker.py")
    with pytest.raises(AssertionError, match="serves --rollout-precision bf16"):
        launcher.ScriptArgs(model_name="mimo26-p4-bf16", rollout_precision=precision)
