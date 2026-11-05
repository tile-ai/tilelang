# DeepSeek V3.2

Install the matching TileLang frontend, TIRx provider and FFI bundle using the
[source installation guide](../../../docs/get_started/Installation.md), then
install this directory's `requirements.txt`. The former `tilelang==0.1.6` pin
describes the original demo environment; it must not replace the selected
compiler bundle when exercising this revision. Model weights and the required
GPU capacity are separate from compiler installation.

First convert huggingface model weights to the format required by our inference demo. Set `MP` to match your available GPU count:
```bash
cd inference
export EXPERTS=256
python convert.py --hf-ckpt-path ${HF_CKPT_PATH} --save-path ${SAVE_PATH} --n-experts ${EXPERTS} --model-parallel ${MP}
```

Launch the interactive chat interface and start exploring DeepSeek's capabilities:
```bash
export CONFIG=config_671B_v3.2.json
torchrun --nproc-per-node ${MP} generate.py --ckpt-path ${SAVE_PATH} --config ${CONFIG} --interactive
```
