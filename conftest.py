# flake8: noqa: E501
import os
import typing as ty
from tempfile import mkdtemp
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
import pytest
import random
import xnat4tests
import medimages4tests.dummy.dicom.mri.fmap.siemens.skyra.syngo_d13c
import medimages4tests.dummy.dicom.mri.dwi.siemens.skyra.syngo_d13c
import medimages4tests.dummy.dicom.mri.t1w.siemens.skyra.syngo_d13c
from frametree.core.frameset import FrameSet
from frametree.core.row import DataRow
from fileformats.medimage import DicomSeries
from frametree.xnat.api import Xnat
from frametree.xnat.testing import (
    TestXnatDatasetBlueprint,
    ScanBlueprint as ScanBP,
)
from frametree.testing.blueprint import FileSetEntryBlueprint as FileBP


PKG_DIR = Path(__file__).parent


# For debugging in IDE's don't catch raised exceptions and let the IDE
# break at it
if os.getenv("_PYTEST_RAISE", "0") != "0":

    @pytest.hookimpl(tryfirst=True)
    def pytest_exception_interact(call: pytest.CallInfo[ty.Any]) -> None:
        if call.excinfo is not None:
            raise call.excinfo.value

    @pytest.hookimpl(tryfirst=True)
    def pytest_internalerror(excinfo: pytest.ExceptionInfo[BaseException]) -> None:
        raise excinfo.value


@pytest.fixture(scope="session")
def run_prefix() -> str:
    "A datetime string used to avoid stale data left over from previous tests"
    return datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")


@pytest.fixture(scope="session")
def frametree_home() -> ty.Generator[Path, None, None]:
    frametree_home = Path(mkdtemp()) / "frametree-home"
    with patch.dict(os.environ, {"FRAMETREE_HOME": str(frametree_home)}):
        yield frametree_home


# -----------------------
# Test dataset structures
# -----------------------


TEST_XNAT_DATASET_BLUEPRINTS = {
    "basic": TestXnatDatasetBlueprint(  # dataset name
        dim_lengths=[1, 1, 1],  # number of visits, groups and members respectively
        scans=[
            ScanBP(
                name="fmap",
                resources=[
                    FileBP(
                        path="DICOM",
                        datatype=DicomSeries,
                        filenames=[f"dicom/fmap/{i}.dcm" for i in range(120)],
                    ),
                ],
            ),
            ScanBP(
                name="t1w",
                resources=[
                    FileBP(
                        path="DICOM",
                        datatype=DicomSeries,
                        filenames=[f"dicom/t1w/{i}.dcm" for i in range(192)],
                    ),
                ],
            ),
            ScanBP(
                name="dwi",
                resources=[
                    FileBP(
                        path="DICOM",
                        datatype=DicomSeries,
                        filenames=[f"dicom/dwi/{i}.dcm" for i in range(4020)],
                    ),
                ],
            ),
        ],
    ),
}

DATASETS = ["basic"]


@pytest.fixture(params=DATASETS, scope="function")
def frameset(
    xnat_repository: Xnat,
    source_data: Path,
    run_prefix: str,
    request: pytest.FixtureRequest,
) -> FrameSet:
    """Creates a dataset that can be mutated (as its name is unique to the function)"""
    dataset_id = request.param
    blueprint = TEST_XNAT_DATASET_BLUEPRINTS[dataset_id]
    project_id = run_prefix + dataset_id + str(hex(random.getrandbits(16)))[2:]
    blueprint.make_dataset(
        dataset_id=project_id,
        store=xnat_repository,
        source_data=source_data,
        name="",
    )
    return xnat_repository.load_frameset(project_id, name="")


@pytest.fixture(scope="function")
def data_row(frameset: FrameSet) -> DataRow:
    return frameset.row(frequency="session", id="visit0group0member0")


@pytest.fixture(scope="session")
def xnat_repository(
    run_prefix: str, frametree_home: str
) -> ty.Generator[Xnat, None, None]:

    xnat4tests.start_xnat()
    config = xnat4tests.Config()

    repository = Xnat(
        server=config.xnat_uri,
        user=config.xnat_user,
        password=config.xnat_password,
        cache_dir=mkdtemp(),
    )

    repository.save(name="testxnat")

    # Stash a project prefix in the repository object
    repository.__annotations__["run_prefix"] = run_prefix

    yield repository


@pytest.fixture(scope="session")
def source_data() -> Path:
    source_data = Path(tempfile.mkdtemp())
    # Create DICOM data
    dicom_dir = source_data / "dicom"
    dicom_dir.mkdir()
    os.symlink(
        medimages4tests.dummy.dicom.mri.fmap.siemens.skyra.syngo_d13c.get_image(),
        dicom_dir / "fmap",
    )
    os.symlink(
        medimages4tests.dummy.dicom.mri.t1w.siemens.skyra.syngo_d13c.get_image(),
        dicom_dir / "t1w",
    )
    os.symlink(
        medimages4tests.dummy.dicom.mri.dwi.siemens.skyra.syngo_d13c.get_image(),
        dicom_dir / "dwi",
    )
    return source_data
