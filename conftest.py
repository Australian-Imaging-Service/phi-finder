# flake8: noqa: E501
import typing as ty
from tempfile import mkdtemp
import tempfile
from datetime import datetime
from pathlib import Path
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


@pytest.fixture(scope="session")
def run_prefix() -> str:
    "A datetime string used to avoid stale data left over from previous tests"
    return datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")


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
                        filenames=[
                            "dicom/fmap/1.dcm",
                            "dicom/fmap/2.dcm",
                            "dicom/fmap/3.dcm",
                        ],
                    ),
                ],
            ),
            ScanBP(
                name="t1w",
                resources=[
                    FileBP(
                        path="DICOM",
                        datatype=DicomSeries,
                        filenames=[
                            "dicom/t1w/1.dcm",
                            "dicom/t1w/2.dcm",
                            "dicom/t1w/3.dcm",
                        ],
                    ),
                ],
            ),
            ScanBP(
                name="dwi",
                resources=[
                    FileBP(
                        path="DICOM",
                        datatype=DicomSeries,
                        filenames=[
                            "dicom/dwi/1.dcm",
                            "dicom/dwi/2.dcm",
                            "dicom/dwi/3.dcm",
                        ],
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
    dataset_id, access_method = request.param.split(".")
    blueprint = TEST_XNAT_DATASET_BLUEPRINTS[dataset_id]
    project_id = run_prefix + dataset_id + str(hex(random.getrandbits(16)))[2:]
    blueprint.make_dataset(
        dataset_id=project_id,
        store=xnat_repository,
        source_data=source_data,
        name="",
    )
    return xnat_repository.load_frameset(project_id, name="")


@pytest.fixture(scope="session")
def data_row(frameset: FrameSet) -> DataRow:
    return frameset.row("session1")


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
    medimages4tests.dummy.dicom.mri.fmap.siemens.skyra.syngo_d13c.get_image(
        out_dir=dicom_dir / "fmap"
    )
    medimages4tests.dummy.dicom.mri.t1w.siemens.skyra.syngo_d13c.get_image(
        out_dir=dicom_dir / "t1w"
    )
    medimages4tests.dummy.dicom.mri.dwi.siemens.skyra.syngo_d13c.get_image(
        out_dir=dicom_dir / "dwi"
    )
    return source_data
