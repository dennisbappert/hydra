# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

import boto3  # type: ignore
from botocore.exceptions import NoCredentialsError, NoRegionError  # type: ignore
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)
from hydra.test_utils.test_utils import chdir_plugin_root, chdir_hydra_root
from pytest import fixture, mark

from hydra_plugins.hydra_sagemaker_launcher.sagemaker_launcher import AmazonSageMakerLauncher  # type: ignore
from hydra_plugins.hydra_sagemaker_launcher._utils import _s3_combine_url, _s3_url_ensure_trailing_slash

temp_remote_dir = "/tmp/hydra_test/"  # nosec
temp_remote_wheel_dir = "/tmp/wheels/"  # nosec
sweep_dir = "tmp_pytest_dir"  # nosec

win_msg = "Ray doesn't support Windows."

aws_not_configured_msg = "AWS credentials not configured correctly. Skipping AWS tests."
try:
    ec2 = boto3.client("ec2")
    ec2.describe_regions()
    aws_not_configured = False
except (NoCredentialsError, NoRegionError):
    aws_not_configured = True

region = os.environ.get("AWS_SAGEMAKER_REGION", "us-west-2")
role_arn = os.environ.get("AWS_SAGEMAKER_ROLE_ARN", "arn:aws:iam::135937774131:**********/***********")
s3_bucket = _s3_url_ensure_trailing_slash(
    os.environ.get("AWS_SAGEMAKER_S3_BUCKET", "s3://********")
)

test_id = uuid.uuid4()
s3_base_path = _s3_url_ensure_trailing_slash(
    _s3_combine_url(s3_bucket, f"hydra-test-{test_id}")
)
s3_wheels_path = _s3_url_ensure_trailing_slash(
    _s3_combine_url(s3_base_path, "wheels")
)

common_overrides = [
    f"hydra.launcher.sagemaker.region={region}",
    f"hydra.launcher.s3_bucket={s3_bucket}",
    f"hydra.launcher.s3_bucket_prefix={s3_base_path}",
    f"+hydra.launcher.sagemaker.estimator_options.role={role_arn}",
    f"+hydra.launcher.sagemaker.channels.wheels={s3_wheels_path}"

    # testing instances
    # f"+hydra.launcher.sagemaker.estimator_options.instance_count=1",
    # f"+hydra.launcher.sagemaker.estimator_options.instance_type=ml.m5.large",
    # f"+hydra.launcher.sagemaker.estimator_options.disable_profiler=True",
    # f"+hydra.launcher.sagemaker.estimator_options.enable_sagemaker_metrics=False",
    # f"+hydra.launcher.sagemaker.estimator_options.framework_version=1.8.1",
    # f"+hydra.launcher.sagemaker.estimator_options.py_version=py3",
    # f"+hydra.launcher.sagemaker.fit_options.job_name='{uuid.uuid4()}'",

    # TODO: add network isolation
    # and enforce isolation in the iam policy
]

launcher_test_suites_overrides = [
    f"hydra.launcher.dry_run=True",
    # f"hydra.launcher.sync_down.source_dir={sweep_dir}/",
    # f"hydra.launcher.sync_down.target_dir={sweep_dir}",
]
launcher_test_suites_overrides.extend(common_overrides)

log = logging.getLogger(__name__)

chdir_plugin_root()

def run_command(commands: str) -> str:
    log.info(f"running: {commands}")
    output = subprocess.getoutput(commands)
    log.info(f"outputs: {output}")
    return output
    

def build_sagemaker_launcher_wheel(tmp_wheel_dir: str) -> str:
    chdir_hydra_root()
    plugin = "hydra_sagemaker_launcher"
    os.chdir(Path("plugins") / plugin)
    log.info(f"Build wheel for {plugin}, save wheel to {tmp_wheel_dir}.")
    run_command(f"python setup.py sdist bdist_wheel && cp dist/*.whl {tmp_wheel_dir}")
    log.info("Download all plugin dependency wheels.")
    run_command(f"pip download . -d {tmp_wheel_dir}")
    plugin_wheel = run_command("ls dist/*.whl").split("/")[-1]
    chdir_hydra_root()
    return plugin_wheel


def build_core_wheel(tmp_wheel_dir: str) -> str:
    chdir_hydra_root()
    run_command(f"python setup.py sdist bdist_wheel && cp dist/*.whl {tmp_wheel_dir}")

    # download dependency wheel for hydra-core
    run_command(f"pip download -r requirements/requirements.txt -d {tmp_wheel_dir}")
    wheel = run_command("ls dist/*.whl").split("/")[-1]
    return wheel

chdir_plugin_root()

@fixture(scope="module")
def manage_wheels() -> None:
    # build all the wheels
    tmpdir = tempfile.mkdtemp()
    core_wheel = build_core_wheel(tmpdir)
    plugin_wheel = build_sagemaker_launcher_wheel(tmpdir)

    log.info(f"S3 wheels path: {s3_wheels_path}")
    run_command(f"aws s3 cp {tmpdir}/{core_wheel} {s3_wheels_path}{core_wheel}")
    run_command(f"aws s3 cp {tmpdir}/{plugin_wheel} {s3_wheels_path}{plugin_wheel}")

    

@mark.skipif(sys.platform.startswith("win"), reason=win_msg)
def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert AmazonSageMakerLauncher.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Launcher)
    ]


@mark.usefixtures("manage_wheels")
@mark.skipif(aws_not_configured, reason=aws_not_configured_msg)
@mark.parametrize(
    "launcher_name, overrides, tmpdir",
    [
        (
            "amazon_sagemaker",
            launcher_test_suites_overrides,
            Path(sweep_dir),
        )
    ],
)
class TestSageMakerLauncher(LauncherTestSuite):
    """
    Run the Launcher test suite on this launcher.
    """

    pass


# @mark.parametrize(
#     "task_launcher_cfg, extra_flags",
#     [
#         (
#             {},
#             [
#                 "-m",
#                 "hydra/launcher=ray",
#                 "hydra/hydra_logging=hydra_debug",
#                 "hydra/job_logging=disabled",
#             ],
#         )
#     ],
# )
# class TestRayLauncherIntegration(IntegrationTestSuite):
#     """
#     Run this launcher through the integration test suite.
#     """

#     pass
