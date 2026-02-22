"""Claude Code agent integration.

Spawns a Claude Code session inside the Docker container via tmux to
autonomously debug unknown SGLang launch failures. The agent receives
structured context (error logs, model profile, configs tried) and works
inside the container where it can edit SGLang source and restart the server.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from porter.analyzer import ModelProfile
from porter.config import (
    AGENT_TIMEOUT,
    ANTHROPIC_BASE_URL,
    CLAUDE_MODEL,
    SGLANG_DEFAULT_PORT,
    SGLANG_HEALTH_ENDPOINT,
)
from porter.docker_manager import DockerManager
from porter.server_config import ServerConfig

log = logging.getLogger(__name__)

TMUX_SESSION = "porter-agent"

AMD_ROCM_PORTING_SKILL_URL = (
    "https://raw.githubusercontent.com/Arist12/AMD-Skills/main/amd-rocm-porting/SKILL.md"
)


@dataclass
class AgentResult:
    success: bool
    logs: str = ""
    fix_description: str = ""
    duration_seconds: float = 0.0


class ClaudeCodeAgent:
    """Spawn and monitor a Claude Code agent for debugging SGLang failures."""

    def __init__(self, docker: DockerManager, container_name: str):
        self.docker = docker
        self.container = container_name

    def run(
        self,
        profile: ModelProfile,
        config: ServerConfig,
        error_logs: str,
        attempted_fixes: list[str],
        timeout: int = AGENT_TIMEOUT,
    ) -> AgentResult:
        """Launch Claude Code agent to debug the failure. Blocks until health or timeout."""
        start = time.time()

        self._install_skill()
        prompt = self._build_prompt(profile, config, error_logs, attempted_fixes)
        self._launch_agent(prompt)

        success = self._wait_for_resolution(timeout)
        duration = time.time() - start

        agent_logs = self._collect_logs()

        if success:
            log.info("Agent resolved the issue in %.0fs", duration)
        else:
            log.warning("Agent timed out after %.0fs", duration)

        self._cleanup()
        return AgentResult(
            success=success, logs=agent_logs,
            fix_description="Agent-applied fix" if success else "Agent timed out",
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _install_skill(self) -> None:
        """Install the AMD ROCm porting skill into the container's Claude config."""
        cmds = [
            "mkdir -p ~/.claude/commands",
            f'curl -fsSL "{AMD_ROCM_PORTING_SKILL_URL}" '
            f'-o ~/.claude/commands/amd-rocm-porting.md',
        ]
        for cmd in cmds:
            self.docker.exec_cmd(self.container, cmd, timeout=30)

    def _build_prompt(
        self,
        profile: ModelProfile,
        config: ServerConfig,
        error_logs: str,
        attempted_fixes: list[str],
    ) -> str:
        fixes_str = "\n".join(f"  - {f}" for f in attempted_fixes) if attempted_fixes else "  None"
        truncated_logs = error_logs[-4000:] if len(error_logs) > 4000 else error_logs

        return f"""You are debugging a SGLang server launch failure on AMD MI355X GPUs.

MODEL: {profile.model_id}
  Type: {profile.model_type}
  Architecture: {', '.join(profile.architectures)}
  Params: {profile.num_params_billion:.1f}B total, {profile.num_active_params_billion:.1f}B active
  Attention: {profile.attention_type} (heads={profile.num_attention_heads}, kv_heads={profile.num_kv_heads}, head_dim={profile.head_dim})
  MoE: {profile.is_moe} ({profile.num_experts} experts, {profile.num_experts_per_token} active)
  Features: MLA={profile.has_mla}, DeltaNet={profile.has_deltanet}, NSA={profile.has_nsa}, MTP={profile.has_mtp}

CURRENT CONFIG:
  {config.build_shell_cmd()}

PREVIOUSLY ATTEMPTED FIXES:
{fixes_str}

ERROR LOGS (last 4000 chars):
{truncated_logs}

TASK:
1. Read the error logs carefully
2. Identify the root cause
3. Apply a fix (edit SGLang source code if needed, or change launch config)
4. Restart the server and verify it responds to health checks at http://localhost:{config.port}/health
5. If the server is healthy, send a test prompt to verify coherent output

Use /amd-rocm-porting skill for ROCm-specific guidance.
The server log is at /tmp/sglang_server.log.
"""

    def _launch_agent(self, prompt: str) -> None:
        """Start Claude Code in a tmux session inside the container."""
        # Kill any existing session
        self.docker.exec_cmd(
            self.container, f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null || true", timeout=10,
        )

        # Write prompt to file to avoid shell quoting issues
        prompt_escaped = prompt.replace("'", "'\\''")
        self.docker.exec_cmd(
            self.container,
            f"cat > /tmp/porter_agent_prompt.txt << 'PORTER_EOF'\n{prompt}\nPORTER_EOF",
            timeout=10,
        )

        agent_script = (
            f"#!/bin/bash\n"
            f"export ANTHROPIC_BASE_URL={ANTHROPIC_BASE_URL}\n"
            f"export ANTHROPIC_API_KEY=not-used\n"
            f"export CLAUDE_MODEL={CLAUDE_MODEL}\n"
            f"claude --dangerously-skip-permissions -p \"$(cat /tmp/porter_agent_prompt.txt)\"\n"
        )
        agent_script_escaped = agent_script.replace("'", "'\\''")
        self.docker.exec_cmd(
            self.container,
            f"echo '{agent_script_escaped}' > /tmp/porter_agent.sh && chmod +x /tmp/porter_agent.sh",
            timeout=10,
        )

        self.docker.exec_cmd(
            self.container,
            f"tmux new-session -d -s {TMUX_SESSION} /tmp/porter_agent.sh",
            timeout=10,
        )

        log.info("Launching Claude Code agent in tmux session '%s'", TMUX_SESSION)

    def _wait_for_resolution(self, timeout: int) -> bool:
        """Poll health endpoint until server is up or timeout."""
        import requests

        url = f"http://localhost:{SGLANG_DEFAULT_PORT}{SGLANG_HEALTH_ENDPOINT}"
        deadline = time.time() + timeout
        poll_interval = 15

        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    return True
            except requests.ConnectionError:
                pass

            # Check if tmux session is still running
            result = self.docker.exec_cmd(
                self.container, f"tmux has-session -t {TMUX_SESSION} 2>/dev/null && echo ALIVE",
                timeout=10,
            )
            if "ALIVE" not in result.stdout:
                log.info("Agent tmux session ended without healthy server")
                # Give a grace period for server startup
                time.sleep(30)
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200:
                        return True
                except requests.ConnectionError:
                    pass
                return False

            time.sleep(poll_interval)
        return False

    def _collect_logs(self) -> str:
        """Capture agent tmux output and server logs."""
        result = self.docker.exec_cmd(
            self.container,
            f"tmux capture-pane -t {TMUX_SESSION} -p 2>/dev/null || echo 'No tmux output'",
            timeout=10,
        )
        agent_output = result.stdout[:5000]

        result = self.docker.exec_cmd(
            self.container, "tail -n 100 /tmp/sglang_server.log 2>/dev/null", timeout=10,
        )
        server_logs = result.stdout[:3000]

        return f"=== Agent Output ===\n{agent_output}\n\n=== Server Logs ===\n{server_logs}"

    def _cleanup(self) -> None:
        self.docker.exec_cmd(
            self.container, f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null || true", timeout=10,
        )
