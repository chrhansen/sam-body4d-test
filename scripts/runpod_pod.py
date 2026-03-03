#!/usr/bin/env python3
"""RunPod Pod helper for SAM-Body4D demo."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE = "https://rest.runpod.io/v1"
LAST_POD_ID_PATH = Path(__file__).resolve().parents[1] / ".runpod_pod_id"
# Use CUDA 12.8 image for broader host-driver compatibility on RunPod.
DEFAULT_IMAGE = "runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204"
DEFAULT_GPU_TYPES = [
    "NVIDIA RTX A6000",
    "NVIDIA A40",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 3090",
]
DEFAULT_BOOTSTRAP_URL = (
    "https://raw.githubusercontent.com/chrhansen/sam-body4d-test/main/scripts/bootstrap_sam_body4d.sh"
)


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_env_pairs(values: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in values:
        if "=" not in pair:
            raise SystemExit(f"Invalid --env value '{pair}'. Expected KEY=VALUE")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise SystemExit(f"Invalid --env key in '{pair}'")
        out[key] = value
    return out


def runpod_request(
    method: str,
    path: str,
    *,
    api_key: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
    timeout_s: int = 120,
) -> Any:
    url = f"{API_BASE}{path}"
    if query:
        clean_query: dict[str, str] = {}
        for key, value in query.items():
            if isinstance(value, bool):
                clean_query[key] = "true" if value else "false"
            else:
                clean_query[key] = str(value)
        url = f"{url}?{urlencode(clean_query)}"

    body = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url=url, data=body, method=method, headers=headers)

    try:
        with urlopen(req, timeout=timeout_s) as response:
            data = response.read().decode("utf-8")
            if not data:
                return None
            return json.loads(data)
    except HTTPError as err:
        body_text = err.read().decode("utf-8", errors="replace").strip() or "<empty body>"
        raise SystemExit(f"RunPod API {method} {path} failed ({err.code}): {body_text}") from err
    except URLError as err:
        raise SystemExit(f"RunPod API {method} {path} failed: {err}") from err


def require_api_key() -> str:
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY is missing. Set it in env or .env")
    return api_key


def save_last_pod_id(pod_id: str) -> None:
    LAST_POD_ID_PATH.write_text(f"{pod_id}\n", encoding="utf-8")


def load_last_pod_id() -> str:
    if not LAST_POD_ID_PATH.exists():
        raise SystemExit("No pod ID found. Pass --pod-id or run create first.")
    pod_id = LAST_POD_ID_PATH.read_text(encoding="utf-8").strip()
    if not pod_id:
        raise SystemExit("Stored pod ID file is empty. Pass --pod-id explicitly.")
    return pod_id


def resolve_pod_id(value: str | None) -> str:
    return value if value else load_last_pod_id()


def proxy_url(pod_id: str, port: int) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


def summarize_pod(pod: dict[str, Any], http_port: int) -> str:
    pod_id = pod.get("id", "<unknown>")
    name = pod.get("name", "<unknown>")
    desired_status = pod.get("desiredStatus", "<unknown>")
    image = pod.get("image", "<unknown>")
    public_ip = pod.get("publicIp")
    port_mappings = pod.get("portMappings") or {}
    mapped_http_port = port_mappings.get(str(http_port))

    lines = [
        f"pod_id: {pod_id}",
        f"name: {name}",
        f"desired_status: {desired_status}",
        f"image: {image}",
        f"proxy_url: {proxy_url(pod_id, http_port)}",
    ]

    if public_ip and mapped_http_port:
        lines.append(f"public_tcp: http://{public_ip}:{mapped_http_port}")
    elif public_ip:
        lines.append(f"public_ip: {public_ip}")

    return "\n".join(lines)


def create_command(args: argparse.Namespace) -> None:
    api_key = require_api_key()

    hf_token = (args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY") or "").strip()
    if not hf_token:
        raise SystemExit("HUGGINGFACE_API_KEY/HF_TOKEN is required for checkpoint downloads")

    gpu_type_ids = split_csv(args.gpu_types)
    if not gpu_type_ids:
        raise SystemExit("At least one GPU type is required")

    pod_name = args.name or f"sam-body4d-{int(time.time())}"

    runtime_env = {
        "HF_TOKEN": hf_token,
        "HUGGINGFACE_API_KEY": hf_token,
        "SAM_BODY4D_REPO": args.sam_body4d_repo,
        "SAM_BODY4D_REF": args.sam_body4d_ref,
        "SAM_BODY4D_WORKDIR": args.workdir,
        "SAM_BODY4D_CKPT_ROOT": args.ckpt_root,
        "SAM_BODY4D_BATCH_SIZE": str(args.batch_size),
        "SAM_BODY4D_ENABLE_COMPLETION": "true" if args.enable_completion else "false",
        "GRADIO_SERVER_NAME": "0.0.0.0",
        "GRADIO_SERVER_PORT": str(args.http_port),
    }
    runtime_env.update(parse_env_pairs(args.env))

    start_cmd = (
        "set -euo pipefail; "
        "mkdir -p /workspace; "
        f"curl -fsSL {args.bootstrap_url} -o /tmp/bootstrap_sam_body4d.sh; "
        "chmod +x /tmp/bootstrap_sam_body4d.sh; "
        "if bash /tmp/bootstrap_sam_body4d.sh 2>&1 | tee /workspace/bootstrap.log; then "
        "  echo 'bootstrap finished'; "
        "else "
        "  echo 'bootstrap failed; serving /workspace on :7860'; "
        "  python -m http.server 7860 --directory /workspace; "
        "fi"
    )

    payload: dict[str, Any] = {
        "name": pod_name,
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "imageName": args.image,
        "gpuCount": args.gpu_count,
        "gpuTypeIds": gpu_type_ids,
        "gpuTypePriority": args.gpu_type_priority,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": args.volume_mount_path,
        "ports": [f"{args.http_port}/http", "22/tcp"],
        "interruptible": args.interruptible,
        "dockerStartCmd": ["bash", "-lc", start_cmd],
        "env": runtime_env,
    }

    if args.data_centers:
        payload["dataCenterIds"] = split_csv(args.data_centers)
        payload["dataCenterPriority"] = args.data_center_priority

    pod = runpod_request("POST", "/pods", api_key=api_key, payload=payload)
    pod_id = pod.get("id")
    if not pod_id:
        raise SystemExit(f"Unexpected create response: {json.dumps(pod, indent=2)}")

    save_last_pod_id(pod_id)

    print("Created pod")
    print(summarize_pod(pod, args.http_port))
    print(f"stored_pod_id_file: {LAST_POD_ID_PATH}")

    if args.wait:
        wait_for_command(
            argparse.Namespace(
                pod_id=pod_id,
                timeout_s=args.wait_timeout_s,
                interval_s=args.wait_interval_s,
                http_port=args.http_port,
            )
        )


def get_pod(pod_id: str) -> dict[str, Any]:
    api_key = require_api_key()
    pod = runpod_request(
        "GET",
        f"/pods/{pod_id}",
        api_key=api_key,
        query={"includeMachine": True, "includeNetworkVolume": True},
    )
    if not isinstance(pod, dict):
        raise SystemExit(f"Unexpected pod response: {pod}")
    return pod


def status_command(args: argparse.Namespace) -> None:
    pod_id = resolve_pod_id(args.pod_id)
    pod = get_pod(pod_id)
    print(summarize_pod(pod, args.http_port))


def wait_for_command(args: argparse.Namespace) -> None:
    pod_id = resolve_pod_id(args.pod_id)
    deadline = time.time() + args.timeout_s

    while True:
        pod = get_pod(pod_id)
        status = pod.get("desiredStatus", "<unknown>")
        now = int(time.time())
        print(f"[{now}] desired_status={status}")

        if status == "RUNNING":
            print("Pod is RUNNING")
            print(summarize_pod(pod, args.http_port))
            return

        if time.time() >= deadline:
            raise SystemExit("Timed out waiting for pod to reach RUNNING state")

        time.sleep(args.interval_s)


def control_command(args: argparse.Namespace, action: str) -> None:
    pod_id = resolve_pod_id(args.pod_id)
    api_key = require_api_key()

    if action == "delete":
        runpod_request("DELETE", f"/pods/{pod_id}", api_key=api_key)
        print(f"Deleted pod {pod_id}")
        if LAST_POD_ID_PATH.exists() and LAST_POD_ID_PATH.read_text(encoding="utf-8").strip() == pod_id:
            LAST_POD_ID_PATH.unlink()
        return

    runpod_request("POST", f"/pods/{pod_id}/{action}", api_key=api_key, payload={})
    print(f"{action} requested for pod {pod_id}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RunPod helper for SAM-Body4D demo")
    p.add_argument("--env-file", default=".env", help="Optional .env path")
    sub = p.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a new pod for SAM-Body4D")
    create.add_argument("--name", default="", help="Pod name (default: autogenerated)")
    create.add_argument("--hf-token", default="", help="HF token override")
    create.add_argument("--image", default=DEFAULT_IMAGE)
    create.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    create.add_argument("--gpu-count", type=int, default=1)
    create.add_argument("--gpu-types", default=",".join(DEFAULT_GPU_TYPES))
    create.add_argument("--gpu-type-priority", choices=["availability", "custom"], default="availability")
    create.add_argument("--data-centers", default="", help="Comma-separated RunPod data center IDs")
    create.add_argument("--data-center-priority", choices=["availability", "custom"], default="availability")
    create.add_argument("--interruptible", action="store_true", help="Create as spot/interruptible pod")
    create.add_argument("--container-disk-gb", type=int, default=80)
    create.add_argument("--volume-gb", type=int, default=120)
    create.add_argument("--volume-mount-path", default="/workspace")
    create.add_argument("--http-port", type=int, default=7860)
    create.add_argument("--bootstrap-url", default=DEFAULT_BOOTSTRAP_URL)
    create.add_argument("--sam-body4d-repo", default="https://github.com/gaomingqi/sam-body4d.git")
    create.add_argument("--sam-body4d-ref", default="master")
    create.add_argument("--workdir", default="/workspace/sam-body4d")
    create.add_argument("--ckpt-root", default="/workspace/checkpoints")
    create.add_argument("--batch-size", type=int, default=16)
    create.add_argument("--enable-completion", action="store_true")
    create.add_argument("--env", action="append", default=[], help="Extra env vars (KEY=VALUE)")
    create.add_argument("--wait", action="store_true", help="Wait until pod is RUNNING")
    create.add_argument("--wait-timeout-s", type=int, default=3600)
    create.add_argument("--wait-interval-s", type=int, default=20)
    create.set_defaults(func=create_command)

    status = sub.add_parser("status", help="Get pod status")
    status.add_argument("--pod-id", default="")
    status.add_argument("--http-port", type=int, default=7860)
    status.set_defaults(func=status_command)

    wait = sub.add_parser("wait", help="Wait until pod is RUNNING")
    wait.add_argument("--pod-id", default="")
    wait.add_argument("--timeout-s", type=int, default=3600)
    wait.add_argument("--interval-s", type=int, default=20)
    wait.add_argument("--http-port", type=int, default=7860)
    wait.set_defaults(func=wait_for_command)

    start = sub.add_parser("start", help="Start a stopped pod")
    start.add_argument("--pod-id", default="")
    start.set_defaults(func=lambda args: control_command(args, "start"))

    stop = sub.add_parser("stop", help="Stop a running pod")
    stop.add_argument("--pod-id", default="")
    stop.set_defaults(func=lambda args: control_command(args, "stop"))

    restart = sub.add_parser("restart", help="Restart a pod")
    restart.add_argument("--pod-id", default="")
    restart.set_defaults(func=lambda args: control_command(args, "restart"))

    delete = sub.add_parser("delete", help="Delete a pod")
    delete.add_argument("--pod-id", default="")
    delete.set_defaults(func=lambda args: control_command(args, "delete"))

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    env_path = Path(args.env_file)
    load_env_file(env_path)

    args.func(args)


if __name__ == "__main__":
    main()
