"""Utilities for validating file structure and preventing duplicates."""

from pathlib import Path
from typing import List, Set, Dict

from clay_ai.core.config import settings


def get_file_hash(file_path: str) -> str:
    """Generate a hash for file content to detect duplicates."""
    with open(file_path, 'rb') as f:
        return str(hash(f.read()))


class DirectoryValidator:
    """Validates directory structure and prevents duplicates."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = Path(root_dir)
        self.file_hashes = dict[str, set[str]]()
        self.scan_directory()

    def scan_directory(self) -> None:
        """Scan directory to build hash map of existing files."""
        for path in self.root_dir.rglob('*'):
            if path.is_file():
                file_hash = get_file_hash(str(path))
                rel_path = str(path.relative_to(self.root_dir))
                if file_hash not in self.file_hashes:
                    self.file_hashes[file_hash] = set()
                self.file_hashes[file_hash].add(rel_path)

    def find_duplicates(self) -> Dict[str, List[str]]:
        """Find duplicate files based on content."""
        duplicates = {}
        for file_hash, paths in self.file_hashes.items():
            if len(paths) > 1:
                duplicates[file_hash] = sorted(list(paths))
        return duplicates

    def is_duplicate(self, file_path: str) -> bool:
        """Check if a file would be a duplicate."""
        abs_path = self.root_dir / file_path
        if not abs_path.exists():
            return False
        file_hash = get_file_hash(str(abs_path))
        return file_hash in self.file_hashes

    def validate_structure(self) -> List[str]:
        """Validate directory structure against expected layout."""
        errors = []
        required_dirs = {'agents', 'api', 'core', 'models', 'tests', 'utils'}
        
        # Check required directories exist
        for dir_name in required_dirs:
            dir_path = self.root_dir / dir_name
            if not dir_path.is_dir():
                errors.append(f"Missing required directory: {dir_name}")

        # Check required files exist
        required_files = {
            'pyproject.toml',
            'README.md',
            'folder_structure.yaml',
            '.cursorrules'
        }
        for file_name in required_files:
            file_path = self.root_dir / file_name
            if not file_path.is_file():
                errors.append(f"Missing required file: {file_name}")

        return errors


# Create singleton instance
validator = DirectoryValidator(str(Path(__file__).parent.parent)) 