from dataclasses import MISSING, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from hydra.core.config_store import ConfigStore

REMOTE_INVOKE_SCRIPT_NAME = "_remote_invoke.py"


@dataclass
class S3SyncConf:
    name: str = MISSING
    source_dir: Optional[str] = MISSING
    target_dir: Optional[str] = None
    s3_prefix: Optional[str] = None
    include: Optional[List[str]] = field(default_factory=list)
    exclude: Optional[List[str]] = field(default_factory=list)


@dataclass
class CommandConf:
    name: str = MISSING
    cmd: List[str] = MISSING
    shell: bool = False
    extend_path: Optional[List[str]] = None


@dataclass
class ScriptConf:
    script_dir: str = MISSING


@dataclass
class AmazonSageMakerConf:
    region: Optional[str] = None

    channels: Dict[str, Any] = field(default_factory=dict)

    estimator_options: Dict[str, Any] = field(
        default_factory=lambda: {"entry_point": REMOTE_INVOKE_SCRIPT_NAME}
    )

    fit_options: Dict[str, Any] = field(
        default_factory=lambda: {
            "wait": True,
            "logs": "All",
            "job_name": None,
            "experiment_config": None,
        }
    )


@dataclass
class AmazonSageMakerLauncherConf:
    _target_: str = (
        "hydra_plugins.hydra_sagemaker_launcher.sagemaker_launcher.AmazonSageMakerLauncher"
    )

    dry_run: bool = False

    s3_bucket: str = ""
    s3_bucket_prefix: str = ""

    script: Optional[ScriptConf] = None

    commands_up: Optional[List[CommandConf]] = None

    # sync_up is executed before launching jobs on the cluster.
    # This can be used for syncing up source code to remote cluster for execution.
    # You need to sync up if your code contains multiple modules.
    # source is local dir, target is remote dir
    sync_up: Optional[List[S3SyncConf]] = None

    sagemaker: AmazonSageMakerConf = AmazonSageMakerConf()

    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)


config_store = ConfigStore.instance()
config_store.store(
    group="hydra/launcher",
    name="amazon_sagemaker",
    node=AmazonSageMakerLauncherConf,
    provider="amazon_sagemaker_launcher",
)
