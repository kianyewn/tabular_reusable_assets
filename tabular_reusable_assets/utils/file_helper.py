import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

import joblib
from loguru import logger


class FileHelperBase:
    """Utility class for parsing file paths."""

    @staticmethod
    def get_absolute_path(filepath: str | Path) -> Path:
        """Get absolute path."""
        return Path(filepath).resolve()

    @staticmethod
    def get_base_name(filepath: str | Path) -> str:
        """Get file name without extension."""
        return Path(filepath).stem

    @staticmethod
    def get_extensions(filepath: str | Path) -> List[str]:
        """Get all extensions."""
        return Path(filepath).suffixes

    @staticmethod
    def get_extension(filepath: str | Path) -> str:
        """Get primary extension."""
        return Path(filepath).suffix

    @classmethod
    def get_parent_dir(cls, filepath) -> Path:
        """Get parent directory."""
        return Path(filepath).parent

    @staticmethod
    def get_path_without_extension(filepath: str | Path) -> Path:
        """Get path without extension."""
        return Path(filepath).with_suffix("")

    @staticmethod
    def get_file_name(filepath: str | Path) -> str:
        """Get file name."""
        return Path(filepath).name

    @staticmethod
    def iterdir(path: str | Path) -> List[Path]:
        """Iterate over directory.
        [PosixPath('/Users/kianyewngieng/github_projects/tabular_reusable_assets/tabular_reusable_assets/metrics'),
        PosixPath('/Users/kianyewngieng/github_projects/tabular_reusable_assets/tabular_reusable_assets/feature_analysis'),
        PosixPath('/Users/kianyewngieng/github_projects/tabular_reusable_assets/tabular_reusable_assets/config'),
        """
        return list(Path(path).resolve().iterdir())

    @classmethod
    def iter_dir_with_ignore(cls, path: str | Path, ignore: List[str]) -> List[Path]:
        """Iterate over directory with ignore."""
        return [p for p in cls.iterdir(path) if p.name not in ignore]

    def get_dir_before_date(file_path, date_regex=r"\d{4}-\d{2}-\d{2}"):
        pattern = rf".*(?={date_regex})"
        # print(pattern)
        match = re.search(pattern, file_path)
        return match.group(0)

    @staticmethod
    def get_latest_file_from_filepaths(file_paths: List, date_regex=r".*(\d{4}-\d{2}-\d{2}/.*)"):
        sorted_files = [
            (match.group(1), match.group(0))
            for match in (re.match(date_regex, str(file)) for file in file_paths)
            if match is not None
        ]
        if len(sorted_files) == 0:
            raise ValueError(f"No files with date_regex `{date_regex}`found.")
        latest_file = sorted(sorted_files, key=lambda x: x[0])[-1][1]
        return latest_file

    @classmethod
    def get_latest_file(cls, filepath: str, date_regex=r".*(\d{4}-\d{2}-\d{2}/.*)"):
        dir_before_date = cls.get_dir_before_date(filepath, date_regex)
        logger.info(dir_before_date)
        # filepaths_with_date = cls.iterdir(dir_before_date)
        filename = cls.get_file_name(filepath)
        filepaths_with_date = Path(dir_before_date).rglob(
            filename
        )  # find all the files that have the filename recursively
        return cls.get_latest_file_from_filepaths(filepaths_with_date, date_regex=date_regex)


class FileHelper(FileHelperBase):
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

    @staticmethod
    def remove_file(filepath: str) -> None:
        filepath = Path(filepath)
        # Remove the file
        if filepath.exists() and filepath.is_file():
            filepath.unlink()
            logger.info(f"Successfully removed file from `{filepath}`.")
        else:
            logger.info("File does not exist.")

    @staticmethod
    def is_parent_dir_exist(filepath) -> bool:
        """Check if parent directory exists."""
        return Path(filepath).parent.exists()

    @staticmethod
    def is_file_exist(filepath) -> bool:
        """Check if file exists."""
        return Path(filepath).exists()

    @staticmethod
    def make_parent_dir_exist_ok(filepath) -> bool:
        return Path(filepath).parent.mkdir(parents=True, exist_ok=True)

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
    ) -> Tuple[List[str], Dict]:
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

    @staticmethod
    def replace_date_in_path(path: str, replacement_date: Optional[str] = None, regex: Optional[str] = None) -> str:
        """Replace date pattern (YYYY-MM-DD) in a file path with a replacement date.

        Args:
            path: File path containing a date pattern
            replacement_date: Date string to replace with. If None, keeps original date.

        Returns:
            Updated file path with replaced date

        Examples:
            >>> replace_date_in_path("data/2025-01-01/train.csv", "2024-12-31")
            'data/2024-12-31/train.csv'

            >>> replace_date_in_path("data/2025-01-01/train.csv")
            'data/2025-01-01/train.csv'
        """
        if replacement_date is None:
            return path

        date_pattern = r"\d{4}-\d{2}-\d{2}" if regex is None else regex
        return re.sub(date_pattern, replacement_date, path)

    @classmethod
    def try_mkdir(cls, dataset_path: str):
        if not cls.is_parent_dir_exist(dataset_path):
            Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Creating parent directory: {cls.get_parent_dir(dataset_path)}")
        return

    @classmethod
    def glob_files(cls, path: str, pattern: str) -> Generator[Path, None, None]:
        for path in Path(path).rglob(pattern):
            yield path

    @staticmethod
    def load_dict(path):
        return joblib.load(path)

    @staticmethod
    def save_dict(data, path):
        return joblib.dump(data, path)
