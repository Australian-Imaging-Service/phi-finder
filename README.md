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

## De-identifying headers with the DICOM PS3.15 profile

The `use_case` argument selects how header values are de-identified:

| `use_case` | Standard headers | Private headers | Patient characteristics |
|------------|------------------|-----------------|-------------------------|
| `"Standard"` (default), `"Aggressive"`, or any other value | Presidio + GLiNER NER redaction | Presidio + GLiNER NER redaction | Sex kept; age → `000Y`; birth date → year |
| `"PS3.15"` / `"dicom_default"` | PS3.15 Basic Profile | Removed | Removed |
| `"PS3.15_Rtn. Pat."` / `"dicom_retain_patient"` | PS3.15 Basic Profile | Removed | Kept (age, sex, size, weight, …) |
| `"dicom_default_scan_private"` | PS3.15 Basic Profile | NER-scrubbed (kept) | Removed |
| `"dicom_retain_patient_scan_private"` | PS3.15 Basic Profile | NER-scrubbed (kept) | Kept (age, sex, size, weight, …) |

Matching is case-insensitive and separator-tolerant (see the note at the end of
this section). Each option is described in detail below.

By default `anonymise_image` scans the header values with the Presidio NER
pipeline (and GLiNER, when supplied). Passing `use_case="PS3.15"` (or its
friendlier alias `use_case="dicom_default"`) instead
de-identifies the headers with the DICOM
[PS3.15 Annex E Basic Application Level Confidentiality Profile](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/chapter_E.html).
This applies the standard's per-attribute actions (empty, dummy, remove, or
remap UIDs), records the de-identification in `DeidentificationMethod` /
`DeidentificationMethodCodeSequence`, and sets `PatientIdentityRemoved` to
`YES`. In this mode the NER engines are **not** run on the headers, so you do
not need to build an `analyser`.

```python
import pydicom as dicom
from phi_finder.dicom_tools import anonymise_dicom

path = "/path/to/some/dicom.dcm"
dcm = dicom.dcmread(path)
anonymised_dcm = anonymise_dicom.anonymise_image(dcm, use_case="PS3.15")
anonymised_dcm.save_as('/path/to/some/dicom_anon.dcm')
```

### Retain Patient Characteristics

Use `use_case="PS3.15_Rtn. Pat."` (or its friendlier alias
`use_case="dicom_retain_patient"`) to apply the basic profile together with the
PS3.15 *Retain Patient Characteristics* Option. Direct identifiers (patient
name, birth date, etc.) are still removed, but patient characteristics such as
age, sex, size, weight, ethnic group and smoking status are kept. The retain
option is recorded in `DeidentificationMethodCodeSequence` (code `113108`).

```python
import pydicom as dicom
from phi_finder.dicom_tools import anonymise_dicom

path = "/path/to/some/dicom.dcm"
dcm = dicom.dcmread(path)
anonymised_dcm = anonymise_dicom.anonymise_image(dcm, use_case="PS3.15_Rtn. Pat.")
anonymised_dcm.save_as('/path/to/some/dicom_anon.dcm')
```

### Scanning private headers

Both DICOM modes have a `_scan_private` variant —
`use_case="dicom_default_scan_private"` and
`use_case="dicom_retain_patient_scan_private"`. The standard headers are handled
exactly as in the matching profile above, but instead of **removing** private
attributes (the Basic Profile default), they are **kept** and their text values
are scanned with the Presidio/GLiNER pipeline. This preserves potentially useful
vendor/clinical private data while still scrubbing any PHI from it. Non-text
private attributes (binary or numeric) are left as-is.

```python
import pydicom as dicom
from phi_finder.dicom_tools import anonymise_dicom

path = "/path/to/some/dicom.dcm"
dcm = dicom.dcmread(path)
anonymised_dcm = anonymise_dicom.anonymise_image(dcm, use_case="dicom_default_scan_private")
anonymised_dcm.save_as('/path/to/some/dicom_anon.dcm')
```

The `use_case` match is case-insensitive and tolerant of separator spelling, so
`"PS3.15"`, `"ps3.15"`, `"PS3_15"`, `"PS3-15"` and the alias `"dicom_default"`
all select the plain profile, and `"PS3.15_Rtn. Pat."`,
`"PS3.15 Retain Patient Characteristics"` or the alias `"dicom_retain_patient"`
select the retain variant. Appending `_scan_private` to either alias selects the
private-header-scanning variant. Any other value (e.g. `"Standard"`, the default,
or `"Aggressive"`) falls back to the Presidio/GLiNER pipeline described above.

> **Note:** `use_case` only controls how the **headers** are handled. Burned-in
> pixel PHI is still redacted only when an `image_redactor` is passed, exactly
> as in the examples above.
