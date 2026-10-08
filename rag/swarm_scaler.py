import time
import shutil
import logging
import subprocess
from contextlib import contextmanager, asynccontextmanager
from typing import Optional
from rag.config import settings

logger = logging.getLogger(__name__)


class DockerSwarmScaler:
    """
    Automates on-demand scaling of the embedding microservice in Docker Swarm:
    - Normal / Idle state: 1 replica
    - Heavy batch ingestion (e.g. >= 50 or 100 chunks): automatically scales up to 3 replicas
    - Automatically scales back down to 1 replica when ingestion completes
    """

    def __init__(
        self,
        service_name: Optional[str] = None,
        scale_threshold: Optional[int] = None,
        scale_up_replicas: Optional[int] = None,
        idle_replicas: Optional[int] = None,
        enabled: Optional[bool] = None
    ):
        self.service_name = service_name or settings.swarm_service_name
        self.scale_threshold = scale_threshold or settings.swarm_scale_threshold
        self.scale_up_replicas = scale_up_replicas or settings.swarm_scale_up_replicas
        self.idle_replicas = idle_replicas or settings.swarm_idle_replicas
        self.enabled = enabled if enabled is not None else settings.auto_scale_swarm

    def is_docker_available(self) -> bool:
        return shutil.which("docker") is not None

    def is_swarm_active(self) -> bool:
        if not self.is_docker_available():
            return False
        try:
            res = subprocess.run(
                ["docker.exe", "info", "--format", "{{.Swarm.LocalNodeState}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return res.returncode == 0 and res.stdout.strip().lower() == "active"
        except Exception:
            return False

    def get_current_replicas(self) -> Optional[int]:
        """
        Returns the number of currently RUNNING tasks for the Swarm service.
        Uses 'docker service ps' filtered to running tasks only.
        """
        if not self.is_swarm_active():
            return None
        try:
            res = subprocess.run(
                [
                    "docker.exe", "service", "ps",
                    "--filter", "desired-state=running",
                    "--format", "{{.CurrentState}}",
                    self.service_name
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if res.returncode == 0:
                lines = [
                    line for line in res.stdout.strip().splitlines()
                    if "running" in line.lower()
                ]
                return len(lines)
        except Exception as e:
            logger.debug(f"Could not inspect service {self.service_name}: {e}")
        return None

    def _issue_scale_command(self, target_replicas: int) -> bool:
        """
        Issues the scale command with --detach=true so the CLI returns immediately
        without waiting for convergence (avoids 15-second timeout on Windows Docker Desktop).
        """
        try:
            res = subprocess.run(
                [
                    "docker.exe", "service", "scale",
                    "--detach=true",
                    f"{self.service_name}={target_replicas}"
                ],
                capture_output=True,
                text=True,
                timeout=10   # Should return almost instantly with --detach=true
            )
            if res.returncode == 0:
                logger.info(f"✓ Scale command issued: '{self.service_name}' → {target_replicas} replica(s)")
                return True
            else:
                logger.warning(
                    f"Scale command failed (exit {res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
                )
                return False
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Scale command timed out even with --detach=true. "
                f"Docker Desktop may be unresponsive."
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to issue scale command: {e}")
            return False

    def _wait_for_replicas(self, target_replicas: int, max_wait: int = 120) -> bool:
        """
        Polls until the running replica count reaches target_replicas or max_wait seconds pass.
        Uses a backoff strategy: polls every 3 seconds.
        """
        logger.info(
            f"⏳ Waiting for {target_replicas} replica(s) to be RUNNING "
            f"(polling every 3s, max {max_wait}s)..."
        )
        start = time.time()
        while time.time() - start < max_wait:
            current = self.get_current_replicas()
            elapsed = int(time.time() - start)
            logger.info(f"   [{elapsed}s] Running replicas: {current} / {target_replicas}")
            if current is not None and current >= target_replicas:
                logger.info(f"✓ All {target_replicas} replica(s) are RUNNING! ({elapsed}s)")
                return True
            time.sleep(3)

        logger.warning(
            f"⚠ Timed out waiting for {target_replicas} replica(s) after {max_wait}s. "
            f"Embedding will continue with however many replicas are available."
        )
        return False

    def scale_service(self, target_replicas: int, wait_for_ready: bool = True) -> bool:
        """
        Scales the Docker Swarm service to target_replicas.
        Uses --detach=true to avoid CLI blocking on Windows, then polls for convergence.
        """
        if not self.is_swarm_active():
            logger.debug("Docker Swarm is not active. Skipping scale operation.")
            return False

        logger.info(
            f"⚡ [Docker Swarm Autoscaler] Scaling '{self.service_name}' "
            f"to {target_replicas} replica(s)..."
        )

        # Issue the non-blocking scale command
        issued = self._issue_scale_command(target_replicas)
        if not issued:
            return False

        if not wait_for_ready:
            return True

        # Scale up: wait for containers to be fully running (extra time for model load)
        max_wait = 120 if target_replicas > 1 else 60
        ready = self._wait_for_replicas(target_replicas, max_wait=max_wait)

        if target_replicas > 1:
            # Extra buffer for Swarm ingress VIP routing table to propagate
            logger.info("⏳ Allowing 5s for Swarm ingress mesh to propagate routes...")
            time.sleep(5)

        return ready

    @contextmanager
    def autoscale(self, chunk_count: int):
        """
        Synchronous context manager that automatically scales up to N replicas for heavy
        chunk loads, and guarantees scaling back down to 1 replica when finished.
        """
        scaled_up = False
        should_scale = (
            self.enabled
            and chunk_count >= self.scale_threshold
            and self.is_swarm_active()
        )

        if should_scale:
            logger.info(
                f"🚀 Ingesting {chunk_count} chunks (threshold: {self.scale_threshold}). "
                f"Autoscaling Swarm service to {self.scale_up_replicas} replicas..."
            )
            scaled_up = self.scale_service(self.scale_up_replicas, wait_for_ready=True)
        else:
            if not self.enabled:
                logger.debug("Auto-scaling disabled.")
            elif chunk_count < self.scale_threshold:
                logger.info(
                    f"ℹ Chunk count {chunk_count} < threshold {self.scale_threshold}. "
                    f"No scale-up needed."
                )

        try:
            yield
        finally:
            if scaled_up and self.is_swarm_active():
                logger.info(
                    f"💤 Ingestion complete. Scaling '{self.service_name}' "
                    f"back to {self.idle_replicas} replica(s)..."
                )
                self.scale_service(self.idle_replicas, wait_for_ready=True)

    @asynccontextmanager
    async def a_autoscale(self, chunk_count: int):
        """
        Asynchronous context manager for async ingestion.
        Delegates to the synchronous scale_service (subprocess calls are blocking by design).
        """
        scaled_up = False
        should_scale = (
            self.enabled
            and chunk_count >= self.scale_threshold
            and self.is_swarm_active()
        )

        if should_scale:
            logger.info(
                f"🚀 [Async] Ingesting {chunk_count} chunks (threshold: {self.scale_threshold}). "
                f"Autoscaling Swarm service to {self.scale_up_replicas} replicas..."
            )
            scaled_up = self.scale_service(self.scale_up_replicas, wait_for_ready=True)

        try:
            yield
        finally:
            if scaled_up and self.is_swarm_active():
                logger.info(
                    f"💤 [Async] Ingestion complete. Scaling '{self.service_name}' "
                    f"back to {self.idle_replicas} replica(s)..."
                )
                self.scale_service(self.idle_replicas, wait_for_ready=True)
