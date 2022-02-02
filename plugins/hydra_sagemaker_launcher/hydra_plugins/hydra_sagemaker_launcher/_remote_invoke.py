import json
import logging
import os
from pathlib import Path

import cloudpickle
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, run_job, setup_globals

log = logging.getLogger(__name__)

JOB_SPEC_PICKLE = "job_spec.pkl"
JOB_RETURN_PICKLE = "returns.pkl"


def _dump_job_return(result: JobReturn, dir: str) -> None:
    path = os.path.join(dir, JOB_RETURN_PICKLE)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        cloudpickle.dump(result, f)
    log.info(f"Pickle for job returns: {f.name}")


def main(jobspec_dir=None) -> JobReturn:
    if "SM_CHANNEL_JOBSPEC" in os.environ and jobspec_dir is None:
        jobspec_dir = os.environ.get("SM_CHANNEL_JOBSPEC")

    with open(os.path.join(jobspec_dir, JOB_SPEC_PICKLE), "rb") as f:
        job_spec = cloudpickle.load(f)
        hydra_context = job_spec["hydra_context"]
        singleton_state = job_spec["singleton_state"]
        sweep_config = job_spec["sweep_config"]
        task_function = job_spec["task_function"]

        if "SM_TRAINING_ENV" in os.environ:
            training_env = json.loads(os.environ.get("SM_TRAINING_ENV"))
            overrides = [f"{k}={v}" for k, v in training_env["hyperparameters"].items()]

            # WIP: this will not for all overrides especially introducing new defaults i guess
            parser = OverridesParser.create()
            parsed_overrides = parser.parse_overrides(overrides=overrides)
            ConfigLoaderImpl._apply_overrides_to_config(parsed_overrides, sweep_config)

        if "SM_MODEL_DIR" in os.environ:
            result_dir = os.environ.get("SM_MODEL_DIR")
            # in SageMaker we only execute one sweep at a time, so we don't need a local subdir within the training container
            sweep_config.hydra.sweep.subdir = ""
        else:
            result_dir = jobspec_dir

        sweep_config.hydra.sweep.dir = result_dir

        setup_globals()
        Singleton.set_state(singleton_state)
        HydraConfig.instance().set_config(sweep_config)

        sweep_dir = Path(str(HydraConfig.get().sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        result = run_job(
            hydra_context=hydra_context,
            task_function=task_function,
            config=sweep_config,
            job_dir_key="hydra.sweep.dir",
            job_subdir_key="hydra.sweep.subdir",
        )

        _dump_job_return(result, result_dir)

        if result.status == JobStatus.FAILED and "SM_OUTPUT_DIR" in os.environ:
            with open(os.path.join(os.environ.get("SM_OUTPUT_DIR"), "failure"), "w") as out_file:
                out_file.write(result.return_value)

        return result


if __name__ == "__main__":
    _ = main()
