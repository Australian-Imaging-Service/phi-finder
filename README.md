# PHI-finder

[![CI/CD](https://github.com/australian-imaging-service/phi-finder/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/australian-imaging-service/phi-finder/actions/workflows/ci-cd.yml)
[![Codecov](https://codecov.io/gh/australian-imaging-service/phi-finder/branch/main/graph/badge.svg?token=UIS0OGPST7)](https://codecov.io/gh/australian-imaging-service/phi-finder)


## Local testing (docker required)

```bash
conda create -n phi-finder python==3.11
conda activate phi-finder
pip install -e .[dev,test] --no-cache-dir
pytest .
```

## Building

```bash
python -m pip install --upgrade build

python -m build

pip install dist/phi_finder-0.1.14-py3-none-any.whl
```

## Basic usage (headers only)


```python
import pydicom as dicom
from phi_finder.dicom_tools import anonymise_dicom

path = "/path/to/some/dicom.dcm"
dcm = dicom.dcmread(path)
anonymised_dcm = anonymise_dicom.anonymise_image(dcm)
anonymised_dcm.save_as('/path/to/some/dicom_anon.dcm')
```

## More advanced usage

```python
import pydicom as dicom
from presidio_image_redactor import (
    DicomImageRedactorEngine, ImageAnalyzerEngine, ContrastSegmentedImageEnhancer)
from phi_finder.dicom_tools import anonymise_dicom

path = "/path/to/some/dicom.dcm"
dcm = dicom.dcmread(path)
score_threshold=.15
analyser = anonymise_dicom._build_presidio_analyser(score_threshold, "en_core_web_lg")
image_redactor = DicomImageRedactorEngine(
    image_analyzer_engine=ImageAnalyzerEngine(
        analyzer_engine=analyser, 
        image_preprocessor=ContrastSegmentedImageEnhancer(),
        ))
anonymised_dcm = anonymise_dicom.anonymise_image(dcm,score_threshold=score_threshold,
                                                 analyser=analyser,
                                                 image_redactor=image_redactor,
                                                 )
anonymised_dcm.save_as('/path/to/some/dicom_anon.dcm')

```
