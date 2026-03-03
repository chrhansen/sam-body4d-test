# SAM-Body4D on RunPod

Deploy helper for running `gaomingqi/sam-body4d` on a RunPod GPU pod.

Default pod image:
- `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04` (matches known-good RunPod host compatibility)

## What this does

- Creates one new RunPod pod for this demo.
- Boots the pod with `scripts/bootstrap_sam_body4d.sh`.
- Installs SAM-Body4D deps.
- Downloads checkpoints with your HF token.
- Launches Gradio app on port `7860`.

Default runtime tuning in bootstrap:
- `sam_3d_body.batch_size=16`
- `completion.enable=false`

This keeps VRAM lower for single-person short clips.

## Required secrets

Set in `.env`:

```bash
RUNPOD_API_KEY=...
HUGGINGFACE_API_KEY=...
```

## Create pod

```bash
python scripts/runpod_pod.py create --wait
```

Optional knobs:

```bash
python scripts/runpod_pod.py create \
  --name sam-body4d-demo \
  --cloud-type SECURE \
  --gpu-types "NVIDIA RTX A6000,NVIDIA A40,NVIDIA GeForce RTX 4090" \
  --batch-size 16 \
  --wait
```

When running, app URL pattern:

```text
https://<pod_id>-7860.proxy.runpod.net
```

## Pod lifecycle

```bash
python scripts/runpod_pod.py status
python scripts/runpod_pod.py stop
python scripts/runpod_pod.py start
python scripts/runpod_pod.py restart
python scripts/runpod_pod.py delete
```

`runpod_pod.py` stores created pod id in `.runpod_pod_id`.
If needed, override with `--pod-id <id>`.

## Upload + run video

In app UI:
1. Upload `.mp4` in left panel.
2. Pick frame, click points on skier (+/-), click `Add Target`.
3. Click `Mask Generation`.
4. Click `4D Generation`.

Outputs land in pod workspace under:

```text
/workspace/sam-body4d/outputs
```
