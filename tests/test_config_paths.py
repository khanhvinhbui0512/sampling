"""Run from the repo root: python tests/test_config_paths.py."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.getcwd())

from src.config import ConfigError, project_path  # noqa: E402


class WorkspacePathsTest(unittest.TestCase):
    """Regression checks for the workspace symlink used by Modal Volumes."""

    def setUp(self):
        self.previous_directory = os.getcwd()
        self.temporary = tempfile.TemporaryDirectory(prefix="qm9-paths-")
        self.repo = os.path.join(self.temporary.name, "repo")
        self.volume = os.path.join(self.temporary.name, "volume")
        os.mkdir(self.repo)
        os.mkdir(self.volume)
        with open(os.path.join(self.repo, "pyproject.toml"), "w", encoding="utf-8"):
            pass
        os.mkdir(os.path.join(self.repo, "src"))
        os.mkdir(os.path.join(self.repo, "resource"))
        os.symlink(self.volume, os.path.join(self.repo, "workspace"))
        os.chdir(self.repo)

    def tearDown(self):
        os.chdir(self.previous_directory)
        self.temporary.cleanup()

    def test_workspace_volume_accepts_relative_artifacts(self):
        path = "workspace/cache/huggingface/datasets"
        self.assertEqual(project_path(path), path)
        self.assertEqual(project_path("src/config.py"), "src/config.py")

    def test_workspace_descendant_cannot_escape_volume(self):
        os.symlink(self.temporary.name, os.path.join(self.volume, "escape"))
        with self.assertRaises(ConfigError):
            project_path("workspace/escape/file.json")

    def test_resource_alias_inside_volume_is_rejected(self):
        os.symlink(os.path.join(self.repo, "resource"), os.path.join(self.volume, "reference"))
        for path in ("workspace/reference/file.json", "resource/file.json"):
            with self.subTest(path=path), self.assertRaises(ConfigError):
                project_path(path)

    def test_non_project_paths_remain_rejected(self):
        for path in ("../outside", "workspace/../../outside", "/tmp/file"):
            with self.subTest(path=path), self.assertRaises(ConfigError):
                project_path(path)


if __name__ == "__main__":
    unittest.main()
