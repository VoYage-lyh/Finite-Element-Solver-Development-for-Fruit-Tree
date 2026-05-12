from __future__ import annotations

import argparse

from orchard_fem.application import OrchardApplication
from orchard_fem.commands.calibrate import register_calibrate_command
from orchard_fem.commands.compare_frf import register_compare_frf_command
from orchard_fem.commands.demo_suite import register_demo_suite_command
from orchard_fem.commands.doctor import register_doctor_command
from orchard_fem.commands.full_validate import register_full_validate_command
from orchard_fem.commands.batch_run import register_batch_run_command
from orchard_fem.commands.harvest import register_harvest_command
from orchard_fem.commands.import_skeleton import register_import_skeleton_command
from orchard_fem.commands.import_topology import register_import_topology_command
from orchard_fem.commands.import_treeqsm import register_import_treeqsm_command
from orchard_fem.commands.modal import register_modal_command
from orchard_fem.commands.plot_frequency_response import register_plot_frequency_response_command
from orchard_fem.commands.plot_time_history import register_plot_time_history_command
from orchard_fem.commands.run import register_run_command
from orchard_fem.commands.verify import register_verify_command
from orchard_fem.commands.visualize import register_visualize_command


def register_all_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    application: OrchardApplication,
) -> None:
    register_run_command(subparsers, application)
    register_batch_run_command(subparsers, application)
    register_modal_command(subparsers, application)
    register_calibrate_command(subparsers, application)
    register_harvest_command(subparsers, application)
    register_compare_frf_command(subparsers, application)
    register_visualize_command(subparsers, application)
    register_plot_frequency_response_command(subparsers, application)
    register_plot_time_history_command(subparsers, application)
    register_import_skeleton_command(subparsers, application)
    register_import_topology_command(subparsers, application)
    register_import_treeqsm_command(subparsers, application)
    register_demo_suite_command(subparsers, application)
    register_verify_command(subparsers, application)
    register_full_validate_command(subparsers, application)
    register_doctor_command(subparsers, application)


__all__ = ["register_all_commands"]
