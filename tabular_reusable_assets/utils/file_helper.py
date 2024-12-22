import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger


class FileHelper:
    IGNORE_FOLDERS = [
        ".DS_Store",
        ".git",
        ".github",
        ".vscode",
        "_pycache_",
        "__pycache__",
    ]
    SPACE = "    "
    BRANCH = "│   "
    TEE = "├── "
    LAST = "└── "

    def __init__(self):
        self.pp = PathParser()

    @staticmethod
    def remove_file(filepath: str) -> None:
        filepath = Path(filepath)
        # Remove the file
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            logger.info(f"Successfully removed file from `{filepath}`.")
        else:
            logger.info("File does not exist.")

    @classmethod
    def print_dir_tree_helper(
        cls,
        root: str | Path,
        level: int = 0,
        current_tree: Optional[List[str]] = list(),
        prefix: str = "",
        dir_tree_metadata: Optional[Dict] = defaultdict(lambda: 0),
        max_depth: int = 100,
        max_dir: int = 100,
    ) -> tuple[List[str], Dict]:
        """
        Helper function to generate directory tree structure.

        Args:
            root: Root directory path
            level: Current directory depth
            current_tree: List to store tree structure
            prefix: Prefix for current line
            dir_tree_metadata: Metadata about directory structure

        Returns:
            Tuple of (tree structure list, directory metadata)
        """

        # https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
        # gs: python tree directory
        # list all files in the directory
        files_and_folders = Path(root).iterdir()  # os.listdir(root)
        sorted_files_and_folders = sorted(
            files_and_folders,
            key=lambda x: (not x.is_dir(), x.name),
        )

        for num, file in enumerate(sorted_files_and_folders):
            if file.name not in cls.IGNORE_FOLDERS:
                current_path = file.resolve()
                pointer = cls.TEE if num < len(sorted_files_and_folders) - 1 else cls.LAST
                if os.path.isdir(current_path):
                    current_tree.append(prefix + pointer + file.name + "/")

                    new_prefix = prefix + cls.BRANCH
                    if max_depth == level or max_dir == dir_tree_metadata["directories"]:
                        pass
                    else:
                        cls.print_dir_tree_helper(
                            root=current_path,
                            level=level + 1,
                            current_tree=current_tree,
                            prefix=new_prefix,
                            dir_tree_metadata=dir_tree_metadata,
                        )
                        dir_tree_metadata["directories"] += 1
                        dir_tree_metadata["level"] = max(dir_tree_metadata["level"], level)
                else:
                    current_tree.append(prefix + pointer + file.name)
                    dir_tree_metadata["files"] += 1
        return current_tree, dir_tree_metadata

    @classmethod
    def print_dir_tree(cls, path, max_depth=100, max_dir=100):
        tree_dir_list, tree_metadata = cls.print_dir_tree_helper(
            root=path,
            current_tree=list(),
            dir_tree_metadata=defaultdict(lambda: 0),
            max_depth=max_depth,
            max_dir=max_dir,
        )
        tree_dir_str = "\n".join([str(path)] + tree_dir_list)
        print(tree_dir_str)
        print(f"Found: {dict(tree_metadata)}")
        return

    def print_dir_tree_simple(self, root, level=0, current_tree=[], indent="    "):
        """Simle solution, print by '- ' delimiter

        Example:
        ```
        - feature_analysis/
            - colors.py
            - exploration.py
        - metrics/
            - __init__.py
            - average_meter.py
            - training_metrics.py
        - model/
            - multi_classification/

        Args:
            root (_type_): _description_
            level (int, optional): _description_. Defaults to 0.
            current_tree (list, optional): _description_. Defaults to [].
            indent (str, optional): _description_. Defaults to '    '.
        ```
        """

        def dfs(root, level=level, current_tree=current_tree, indent=indent):
            # list all files in the directory
            files_and_folders = os.listdir(root)
            sorted_files_and_folders = sorted(
                files_and_folders,
                key=lambda x: (not os.path.isdir(os.path.join(root, x)), x),
            )
            # return sorted_files, os.listdir(root)
            for num, file in enumerate(sorted_files_and_folders):
                if file not in self.IGNORE_FOLDERS:
                    current_path = os.path.join(root, file)
                    if os.path.isdir(current_path):
                        current_tree.append(indent * level + "- " + file + "/")
                    dfs(root=current_path, level=level + 1, current_tree=current_tree)
                else:
                    current_tree.append(indent * level + "- " + file)
            return current_tree

        print("\n".join(dfs(Path(root))))
        return


class PathParser:
    """Utility class for parsing file paths."""

    @staticmethod
    def absolute_path(filepath: str | Path) -> Path:
        """Get absolute path."""
        return filepath.resolve()

    @staticmethod
    def base_name(filepath: str | Path) -> str:
        """Get file name without extension."""
        return filepath.stem

    @staticmethod
    def extensions(filepath: str | Path) -> List[str]:
        """Get all extensions."""
        return filepath.suffixes

    @staticmethod
    def extension(filepath: str | Path) -> str:
        """Get primary extension."""
        return filepath.suffix

    @property
    def directory(filepath) -> Path:
        """Get parent directory."""
        return filepath.parent

    @staticmethod
    def path_without_extension(filepath: str | Path) -> Path:
        """Get path without extension."""
        return filepath.with_suffix("")
