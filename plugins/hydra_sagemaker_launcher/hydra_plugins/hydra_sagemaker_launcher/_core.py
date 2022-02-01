import logging
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

import boto3
import cloudpickle
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, JobStatus, configure_log, filter_overrides, setup_globals
from omegaconf import OmegaConf, open_dict
from sagemaker.pytorch import PyTorch

from hydra_plugins.hydra_sagemaker_launcher._remote_invoke import JOB_RETURN_PICKLE
from hydra_plugins.hydra_sagemaker_launcher._utils import (
    JOB_SPEC_S3_PREFIX,
    Job,
    _describe_job,
    _get_abs_code_dir,
    _job_status_to_str,
    _package_code,
    _parse_job_status,
    _pickle_job,
    _run_command,
    _run_commands,
    _run_sync,
    _s3_combine_url,
    _s3_url_ensure_trailing_slash,
    _split_s3_path,
)
from hydra_plugins.hydra_sagemaker_launcher.sagemaker_launcher import (
    AmazonSageMakerLauncher,  # type: ignore
)

log = logging.getLogger(__name__)


def launch(
    launcher: AmazonSageMakerLauncher,
    job_overrides: Sequence[Sequence[str]],
    initial_job_idx: int,
) -> Sequence[JobReturn]:
    setup_globals()
    assert launcher.config is not None
    assert launcher.hydra_context is not None
    assert launcher.task_function is not None

    configure_log(launcher.config.hydra.hydra_logging, launcher.config.hydra.verbose)

    # we need the s3 client later, it may make sense to initialize it globally to prevent
    # creating it for every sweep
    s3_client = boto3.client("s3", region_name=launcher.sagemaker_cfg.region)

    log.info(f"AWS SageMaker Launcher is launching {len(job_overrides)} jobs")

    s3_base_path = _s3_url_ensure_trailing_slash(
        _s3_combine_url(launcher.s3_bucket, launcher.s3_bucket_prefix)
    )

    log.info(f"Base directory for this launch is {s3_base_path}")

    job_queue: List[Job] = []

    with tempfile.TemporaryDirectory() as local_tmp_dir:
        for idx, overrides in enumerate(job_overrides):
            idx = initial_job_idx + idx
            ostr = " ".join(filter_overrides(overrides))
            log.info(f"\t#{idx} : {ostr}")
            sweep_config = launcher.hydra_context.config_loader.load_sweep_config(
                launcher.config, list(overrides)
            )
            with open_dict(sweep_config):
                sweep_config.hydra.job.num = idx

            _pickle_job(
                tmp_dir=local_tmp_dir,
                hydra_context=launcher.hydra_context,
                sweep_config=sweep_config,
                task_function=launcher.task_function,
                singleton_state=Singleton.get_state(),
            )

            if launcher.dry_run:
                log.info("Performing a dry run by directly executing _remote_invoke.py")
                from ._remote_invoke import main as launch_job_locally

                result = launch_job_locally(local_tmp_dir)
                job_queue.append(Job(sweep_config=sweep_config, result=result))

                # We do not need to process further in a dry_run
                continue

            s3_sweep_dir = _s3_url_ensure_trailing_slash(_s3_combine_url(s3_base_path, str(idx)))

            if launcher.script:
                s3_script_path = _package_code(launcher.script, s3_base=s3_sweep_dir)
                source_dir_estimator_options = {"source_dir": s3_script_path}
            else:
                source_dir_estimator_options = {"source_dir": _get_abs_code_dir(".")}

            if launcher.commands_up is not None and any(launcher.commands_up):
                _run_commands(launcher.commands_up)

            if launcher.sync_up is not None and any(launcher.sync_up):
                _run_sync(launcher.sync_up, s3_base=s3_sweep_dir)

            jobspec_target_dir = _s3_url_ensure_trailing_slash(
                _s3_combine_url(s3_sweep_dir, JOB_SPEC_S3_PREFIX)
            )
            log.info(f"Synchronizing job definition to target directory {jobspec_target_dir}")
            _run_command(
                ["aws", "s3", "sync", local_tmp_dir, jobspec_target_dir],
                shell=False,
                env=os.environ,
                cwd=None,
            )

            estimator = PyTorch(
                **{
                    **source_dir_estimator_options,
                    "output_path": s3_sweep_dir,
                    **OmegaConf.to_container(launcher.sagemaker_cfg.estimator_options),
                }
            )

            log.info(f"Starting training job {jobspec_target_dir}")
            estimator.fit(
                inputs={"jobspec": jobspec_target_dir, **launcher.sagemaker_cfg.channels},
                **launcher.sagemaker_cfg.fit_options,
            )

            job_queue.append(
                Job(training_job=estimator.latest_training_job, sweep_config=sweep_config)
            )

        while True:
            log_message = "(Working...) Status:"
            for idx, current_job in enumerate(job_queue):
                status = _parse_job_status(current_job)

                if status == JobStatus.UNKNOWN and current_job.training_job is not None:
                    current_job._last_training_job_description = _describe_job(
                        current_job.training_job
                    )

                if (
                    status == JobStatus.COMPLETED or status == JobStatus.FAILED
                ) and current_job.result is None:
                    assert (
                        "ModelArtifacts" in current_job._last_training_job_description
                    ), "Every completed training job should have ModelArtifacts."
                    model_artifacts_s3_url = current_job._last_training_job_description[
                        "ModelArtifacts"
                    ]["S3ModelArtifacts"]
                    log.info(
                        f'Downloading model artifacts for training job {current_job._last_training_job_description["TrainingJobName"]} from {model_artifacts_s3_url}'
                    )

                    # noinspection PyBroadException
                    try:
                        with tempfile.TemporaryDirectory() as artifacts_tmp_dir:
                            with open(
                                os.path.join(artifacts_tmp_dir, "artifacts.tar.gz"), "wb"
                            ) as f:
                                bucket, key = _split_s3_path(model_artifacts_s3_url)
                                s3_client.download_fileobj(bucket, key, f)

                            local_sweep_dir = os.path.join(
                                current_job.sweep_config.hydra.sweep.dir,
                                current_job.sweep_config.hydra.sweep.subdir,
                            )
                            local_sweep_dir = Path(_get_abs_code_dir(local_sweep_dir))
                            local_sweep_dir.mkdir(parents=True, exist_ok=True)

                            with tarfile.open(
                                os.path.join(artifacts_tmp_dir, "artifacts.tar.gz")
                            ) as tar:
                                tar.extractall(local_sweep_dir)

                            with open(os.path.join(local_sweep_dir, JOB_RETURN_PICKLE), "rb") as f:
                                job_return = cloudpickle.load(f)  # nosec
                                current_job.result = job_return

                    except:  # noqa: E722
                        error = sys.exc_info()[0]
                        log.error(f"Unable to fetch results from job {idx}: {error}")
                        current_job.result = JobReturn()
                        current_job.result.stats = JobStatus.FAILED
                        current_job.result.return_value = error

                log_message += f" # Job {idx}: {_job_status_to_str(status)}"

            log_message += " #"  # making the output just lookin a bit symmetrical.
            log.info(log_message)

            if all([current_job.result is not None for current_job in job_queue]):
                break

            time.sleep(10)

        return [job.result for job in job_queue]
