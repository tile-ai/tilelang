import os
import subprocess
from typing import Union

# Get the absolute path of the current Python script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the absolute path of the project root directory (one level above the current directory)
develop_project_root_dir = os.path.abspath(os.path.join(current_dir, ".."))
installed_project_root_dir = os.path.abspath(os.path.join(current_dir))
# Define the path to the VERSION file located in the project root directory
develop_version_file_path = os.path.join(develop_project_root_dir, "VERSION")
installed_version_file_path = os.path.join(installed_project_root_dir, "VERSION")

if os.path.exists(develop_version_file_path):
    version_file_path = develop_version_file_path
elif os.path.exists(installed_version_file_path):
    version_file_path = installed_version_file_path
else:
    raise FileNotFoundError("VERSION file not found in the project root directory")

# Read and store the version information from the VERSION file
# Use 'strip()' to remove any leading/trailing whitespace or newline characters
with open(version_file_path, "r") as version_file:
    __version__ = version_file.read().strip()


def get_git_commit_id() -> Union[str, None]:
    """Get the current git commit hash by running git in the current file's directory."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL,
                                       encoding='utf-8').strip()
    # FileNotFoundError is raised when git is not installed
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


# Append git commit hash to version if not already present
# NOTE(lei): Although the local commit id cannot capture locally staged changes,
# the local commit id can help mitigate issues caused by incorrect cache to some extent,
# so it should still be kept.
if "+" not in __version__ and (commit_id := get_git_commit_id()):
    __version__ = f"{__version__}+{commit_id}"

# Define the public API for the module
__all__ = ["__version__"]
