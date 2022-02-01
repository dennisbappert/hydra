from typing import List, Optional, Sequence

from hydra.core.utils import JobReturn
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig

from hydra_plugins.hydra_sagemaker_launcher._config import (  # type: ignore
    CommandConf,
    S3SyncConf,
    ScriptConf,
)


class AmazonSageMakerLauncher(Launcher):
    def __init__(
        self,
        sagemaker: DictConfig,
        dry_run: bool,
        s3_bucket: str,
        s3_bucket_prefix: str,
        script: ScriptConf,
        commands_up: List[CommandConf],
        sync_up: List[S3SyncConf],
        **kwargs,
    ) -> None:
        self.sagemaker_cfg = sagemaker
        self.dry_run = dry_run
        self.s3_bucket = s3_bucket
        self.s3_bucket_prefix = s3_bucket_prefix
        self.script = script
        self.commands_up = commands_up
        self.sync_up = sync_up
        self.hydra_context: Optional[HydraContext] = None
        self.task_function: Optional[TaskFunction] = None
        self.config: Optional[DictConfig] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        from . import _core

        return _core.launch(
            launcher=self, job_overrides=job_overrides, initial_job_idx=initial_job_idx
        )
