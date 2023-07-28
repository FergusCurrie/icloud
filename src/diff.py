"""
Quick file to diff the data base
"""

from pathlib import Path


def diff_directories(directory1: Path, directory2: Path) -> list[str]:
    files1 = set([str(x.stem) for x in directory1.iterdir()])
    files2 = set([str(x.stem) for x in directory2.iterdir()])
    return list(files1 - files2)
