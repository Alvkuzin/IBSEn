"""Pytest options, loadable before argument parsing with ``-p``."""


def pytest_configure(config):
    # Installed tests also work without the repository's pyproject.toml.
    config.addinivalue_line(
        "markers", "gg_direct: optional precise gamma-gamma integration diagnostics",
    )


def pytest_addoption(parser):
    parser.addoption("--run-gg-direct", action="store_true", default=False,
                     help="Run precise tau_gg_iso_2d comparisons and print discrepancies")
