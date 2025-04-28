from pathlib import Path
from frametree.core.row import DataRow


def test_data_row_access(tmp_path: Path, data_row: DataRow) -> None:

    assert len(data_row.entries) == 3
