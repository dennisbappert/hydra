# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys

import boto3  # type: ignore
from botocore.exceptions import NoCredentialsError, NoRegionError  # type: ignore
from hydra.core.plugins import Plugins
from hydra.plugins.launcher import Launcher
from hydra.test_utils.launcher_common_tests import (
    IntegrationTestSuite,
    LauncherTestSuite,
)
from hydra.test_utils.test_utils import chdir_plugin_root
from pytest import mark

from hydra_plugins.hydra_sagemaker_launcher.sagemaker_launcher import AmazonSageMakerLauncher  # type: ignore

temp_remote_dir = "/tmp/hydra_test/"  # nosec
temp_remote_wheel_dir = "/tmp/wheels/"  # nosec
sweep_dir = "tmp_pytest_dir"  # nosec

aws_not_configured_msg = "AWS credentials not configured correctly. Skipping AWS tests."
try:
    ec2 = boto3.client("ec2")
    ec2.describe_regions()
    aws_not_configured = False
except (NoCredentialsError, NoRegionError):
    aws_not_configured = True

chdir_plugin_root()

def test_discovery() -> None:
    # Tests that this plugin can be discovered via the plugins subsystem when looking for Launchers
    assert AmazonSageMakerLauncher.__name__ in [
        x.__name__ for x in Plugins.instance().discover(Launcher)
    ]


@mark.skipif(aws_not_configured, reason=aws_not_configured_msg)
@mark.parametrize("launcher_name, overrides", [("amazon_sagemaker", [])])
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
