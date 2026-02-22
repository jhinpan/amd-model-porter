"""Docker container lifecycle management for SGLang ROCm containers."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

from porter.config import (
    DEFAULT_DOCKER_IMAGE,
    DOCKER_RUN_DEFAULTS,
    MODEL_WEIGHTS_HOST_DIR,
    MODEL_WEIGHTS_CONTAINER_DIR,
    SETUP_GIST_URL,
    SGLANG_DEFAULT_PORT,
    SGLANG_HEALTH_ENDPOINT,
    SGLANG_HEALTH_TIMEOUT,
)
from porter.server_config import ServerConfig

log = logging.getLogger(__name__)


@dataclass
class ContainerInfo:
    name: str
    container_id: str = ""
    image: str = ""
    status: str = "unknown"
    port: int = SGLANG_DEFAULT_PORT


class DockerManager:
    """Manage SGLang ROCm Docker containers on the local machine."""

    def __init__(
        self,
        image: str = DEFAULT_DOCKER_IMAGE,
        weights_dir: str = MODEL_WEIGHTS_HOST_DIR,
        setup_gist: str = SETUP_GIST_URL,
    ):
        self.image = image
        self.weights_dir = weights_dir
        self.setup_gist = setup_gist

    # ------------------------------------------------------------------
    # Shell helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(cmd: str, timeout: int = 600, check: bool = True) -> subprocess.CompletedProcess:
        log.debug("$ %s", cmd)
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        if check and result.returncode != 0:
            log.error("Command failed (exit %d): %s\nstderr: %s", result.returncode, cmd, result.stderr[:2000])
        return result

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def pull_image(self) -> bool:
        log.info("Pulling Docker image: %s", self.image)
        result = self._run(f"docker pull {self.image}", timeout=1200, check=False)
        if result.returncode != 0:
            log.error("Failed to pull image: %s", result.stderr[:500])
            return False
        log.info("Image pulled successfully")
        return True

    def image_exists(self) -> bool:
        result = self._run(f"docker image inspect {self.image}", check=False)
        return result.returncode == 0

    # ------------------------------------------------------------------
    # Container lifecycle
    # ------------------------------------------------------------------

    def create_container(self, name: str) -> ContainerInfo:
        """Create and start a long-running container with dev tools."""
        self.remove_container(name)

        d = DOCKER_RUN_DEFAULTS
        cmd_parts = [
            "docker run",
            f"--cap-add={d['cap_add']}",
            f"--ipc={d['ipc']}",
            f"--privileged={d['privileged']}",
            f"--shm-size={d['shm_size']}",
            f"--network={d['network']}",
            f"--name={name}",
        ]
        for dev in d["devices"]:
            cmd_parts.append(f"--device={dev}")
        cmd_parts.append(f"--group-add {d['group_add']}")
        cmd_parts.append(f"-v {self.weights_dir}:{MODEL_WEIGHTS_CONTAINER_DIR}")
        cmd_parts.append(f"-d {self.image}")

        init_cmd = (
            f'bash -lc "curl -fsSL {self.setup_gist} | bash'
            f' && while true; do sleep 3600; done"'
        )
        cmd_parts.append(init_cmd)

        full_cmd = " \\\n  ".join(cmd_parts)
        log.info("Creating container %s", name)
        result = self._run(full_cmd, timeout=300, check=False)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create container: {result.stderr[:500]}")

        container_id = result.stdout.strip()[:12]
        log.info("Container %s created: %s", name, container_id)

        return ContainerInfo(
            name=name, container_id=container_id, image=self.image, status="running",
        )

    def remove_container(self, name: str) -> None:
        self._run(f"docker rm -f {name}", check=False)

    def stop_container(self, name: str) -> None:
        self._run(f"docker stop {name}", timeout=30, check=False)

    def container_running(self, name: str) -> bool:
        result = self._run(
            f"docker inspect -f '{{{{.State.Running}}}}' {name}", check=False,
        )
        return result.stdout.strip() == "true"

    # ------------------------------------------------------------------
    # Exec commands inside container
    # ------------------------------------------------------------------

    def exec_cmd(
        self,
        name: str,
        cmd: str,
        timeout: int = 600,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Run a command inside the container via docker exec."""
        env_flags = ""
        if env:
            env_flags = " ".join(f"-e {k}={shlex.quote(v)}" for k, v in env.items())
        full = f"docker exec {env_flags} {name} bash -lc {shlex.quote(cmd)}"
        return self._run(full, timeout=timeout, check=False)

    def exec_background(
        self,
        name: str,
        cmd: str,
        env: Optional[dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        """Start a background process inside the container via a temp script."""
        # Write the command to a script to avoid nested quoting issues
        script = f"#!/bin/bash\n{cmd}\n"
        escaped_script = script.replace("'", "'\\''")
        self.exec_cmd(name, f"echo '{escaped_script}' > /tmp/porter_run.sh && chmod +x /tmp/porter_run.sh", timeout=10)

        env_flags = ""
        if env:
            env_flags = " ".join(f"-e {k}={shlex.quote(v)}" for k, v in env.items())
        full = f"docker exec -d {env_flags} {name} bash -c 'nohup /tmp/porter_run.sh > /tmp/sglang_server.log 2>&1'"
        return self._run(full, timeout=30, check=False)

    def get_logs(self, name: str, tail: int = 200) -> str:
        """Get the SGLang server logs from inside the container."""
        result = self.exec_cmd(name, f"tail -n {tail} /tmp/sglang_server.log", timeout=10)
        return result.stdout if result.returncode == 0 else result.stderr

    def kill_server(self, name: str) -> None:
        """Kill any running SGLang server inside the container."""
        self.exec_cmd(name, "pkill -f 'sglang.launch_server' || true", timeout=10)
        time.sleep(2)

    # ------------------------------------------------------------------
    # SGLang server lifecycle
    # ------------------------------------------------------------------

    def launch_server(self, name: str, config: ServerConfig) -> bool:
        """Launch SGLang server inside a container and return True if healthy."""
        self.kill_server(name)
        time.sleep(2)

        shell_cmd = config.build_shell_cmd()
        log.info("Launching SGLang: %s", config.summary())
        self.exec_background(name, shell_cmd)
        return self.wait_for_health(config.port, container_name=name)

    def wait_for_health(
        self,
        port: int = SGLANG_DEFAULT_PORT,
        timeout: int = SGLANG_HEALTH_TIMEOUT,
        container_name: str = "",
    ) -> bool:
        """Poll the SGLang health endpoint until ready or timeout."""
        url = f"http://localhost:{port}{SGLANG_HEALTH_ENDPOINT}"
        deadline = time.time() + timeout
        interval = 5

        log.info("Waiting for SGLang health at %s (timeout=%ds)", url, timeout)
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    log.info("SGLang server is healthy")
                    return True
            except requests.ConnectionError:
                pass

            # Early exit: check if server process crashed
            if container_name:
                check = self.exec_cmd(
                    container_name,
                    "pgrep -f 'python.*sglang' > /dev/null && echo ALIVE || echo DEAD",
                    timeout=5,
                )
                if "DEAD" in check.stdout:
                    log.error("SGLang server process is not running — early exit")
                    return False

            time.sleep(interval)
            remaining = int(deadline - time.time())
            if remaining > 0 and remaining % 30 == 0:
                log.info("Still waiting... %ds remaining", remaining)

        log.error("SGLang health check timed out after %ds", timeout)
        return False

    # ------------------------------------------------------------------
    # Correctness check
    # ------------------------------------------------------------------

    def verify_correctness(self, port: int = SGLANG_DEFAULT_PORT) -> tuple[bool, str]:
        """Send a test prompt and verify coherent output."""
        url = f"http://localhost:{port}/v1/chat/completions"
        payload = {
            "model": "test",
            "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
            "max_tokens": 32,
            "temperature": 0.0,
        }
        try:
            r = requests.post(url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            # Basic sanity: response should be non-empty and not garbled
            if not content or len(content) < 1:
                return False, "Empty response"
            garbage_indicators = ["NaN", "\x00", "ÿÿÿ", "..sp.ER."]
            for g in garbage_indicators:
                if g in content:
                    return False, f"Garbage detected in output: {content[:200]}"
            log.info("Correctness check passed: %s", content[:100])
            return True, content
        except Exception as e:
            return False, str(e)

    # ------------------------------------------------------------------
    # Model weight management
    # ------------------------------------------------------------------

    def ensure_model_weights(self, model_id: str, container_name: str) -> str:
        """Check if model weights exist; if not, download them inside the container."""
        model_name = model_id.split("/")[-1]
        container_path = f"{MODEL_WEIGHTS_CONTAINER_DIR}/{model_name}"

        result = self.exec_cmd(
            container_name, f"test -d {container_path} && echo EXISTS", timeout=10,
        )
        if "EXISTS" in result.stdout:
            log.info("Model weights found at %s", container_path)
            return container_path

        # Check HF hub cache format
        hub_name = f"models--{model_id.replace('/', '--')}"
        hub_path = f"{MODEL_WEIGHTS_CONTAINER_DIR}/hub/{hub_name}"
        result = self.exec_cmd(
            container_name, f"test -d {hub_path} && echo EXISTS", timeout=10,
        )
        if "EXISTS" in result.stdout:
            log.info("Model weights found in HF hub cache: %s", hub_path)
            return model_id  # SGLang can resolve via HF cache

        log.info("Downloading model weights for %s", model_id)
        dl_cmd = (
            f"pip install -U huggingface_hub && "
            f"HF_HUB_CACHE={MODEL_WEIGHTS_CONTAINER_DIR}/hub "
            f"huggingface-cli download {model_id}"
        )
        result = self.exec_cmd(container_name, dl_cmd, timeout=7200)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to download model: {result.stderr[:500]}")
        return model_id
