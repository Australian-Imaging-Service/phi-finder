import pytest
import pydicom
import json
from pydicom.data import get_testdata_files
from pydicom.valuerep import PersonName


from phi_finder.dicom_tools import anonymise_dicom


def test_anonymise_with_transformer():
    model = anonymise_dicom._build_transformer()
    text = "John Doe was born on 01/01/1980 and has a social security number of 123-45-6789."
    anon = anonymise_dicom._anonymise_with_transformer(model, text=text, threshold=0.01)
    assert anon != text
    assert "XXXX" in anon


def test_destroy_pixels():
    filename = get_testdata_files("CT_small.dcm")[0]
    dataset = pydicom.dcmread(filename)
    assert "PixelData" in dataset
    assert dataset.pixel_array.shape != (8, 8)
    assert all(v != 0 for v in dataset.pixel_array.flatten())
    anonymised_dataset = anonymise_dicom.destroy_pixels(dataset)
    assert anonymised_dataset.pixel_array.shape == (8, 8)
    assert all(v == 0 for v in anonymised_dataset.pixel_array.flatten())


TEST_STRINGS_PII = ["John Doe",
                    "Jane Smith",
                    "Female",
                    "Male",
                    "01/01/1980",
                    "F", "M", "19430617", "076Y"]
@pytest.mark.parametrize("test_string", TEST_STRINGS_PII)
def test_presidio_regex_sensitive(test_string: str):
    analyser = anonymise_dicom._build_presidio_analyser(0.5)
    anonymizer = anonymise_dicom.AnonymizerEngine()
    analyzer_results = analyser.analyze(
                    text=test_string, language="en", score_threshold=0.5
                )
    anonymized_text = anonymizer.anonymize(
        text=test_string,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": anonymise_dicom.OperatorConfig("replace", {"new_value": "[XXXX]"})
        },
    ).text
    assert "[XXXX]" in anonymized_text


TEST_STRINGS_CLEAN = ["Not sensitive", "Flat tire", "Most common"]
@pytest.mark.parametrize("test_string", TEST_STRINGS_CLEAN)
def test_presidio_regex_clean(test_string: str):
    analyser = anonymise_dicom._build_presidio_analyser(0.5)
    anonymizer = anonymise_dicom.AnonymizerEngine()
    analyzer_results = analyser.analyze(
                    text=test_string, language="en", score_threshold=0.5
                )
    anonymized_text = anonymizer.anonymize(
        text=test_string,
        analyzer_results=analyzer_results,
        operators={
            "DEFAULT": anonymise_dicom.OperatorConfig("replace", {"new_value": "[XXXX]"})
        },
    ).text
    assert test_string == anonymized_text


def test_anonymise_image():
    dataset = pydicom.dcmread(get_testdata_files("CT_small.dcm")[0])
    #dataset = pydicom.dcmread("0.dcm")
    anonymised_dataset = anonymise_dicom.anonymise_image(dataset,
                                                         analyser=None,
                                                         anonymizer=None,
                                                         image_redactor=None,
                                                         score_threshold=0.5,
                                                         use_transformers=False)
    assert anonymised_dataset.PatientName == PersonName('XXXX')
    assert anonymised_dataset[0x0010, 0x0040].value != 'XXXX'  # Sex unchanged
    if anonymised_dataset[0x0010, 0x0030].value != '':
        assert anonymised_dataset[0x0010, 0x0030].value[4:] == '0101'  # Month and day fixed.
    else:
        assert anonymised_dataset[0x0010, 0x0030].value == ''  # Birthdate empty if not kept
    #assert anonymised_dataset[0x0008, 0x0020].value == '20040119'  # Study Date unchanged
    assert anonymised_dataset[0x0010, 0x1010].value == '000Y'  # Unchanged
    assert (0x02091000) in anonymised_dataset
    assert anonymised_dataset[0x0209, 0x1000].name == '[Flagged Headers PHI-Finder]'
    assert anonymised_dataset[0x0209, 0x1000].VR == 'ST'
    assert json.loads(anonymised_dataset[0x0209, 0x1000].value)
