import os

import pandas as pd
import pydicom as dicom

from presidio_image_redactor import DicomImageRedactorEngine
from presidio_anonymizer import AnonymizerEngine
from pydicom.pixel_data_handlers.util import apply_voi_lut
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


def _parse_value(value):
    if isinstance(value, list):
        return value
    try:
        value = int(value)
    except:
        try:
            value = float(value)
        except:
            value = str(value)[0:1000]
            value = value.replace("\n", " ").replace("\r", " ").replace("^", " ")
    return value


def _build_analyser():
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_md"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    title_recognizer = PatternRecognizer(
        supported_entity="TITLE",
        deny_list=[
            "Dr",
            "DR",
            "Prof",
            "PROF",
            "Prof.",
            "Doctor",
            "DOCTOR",
            "Professor",
            "PROFESSOR",
            "Associate Professor",
            "ASSOCIATE PROF",
            "ASSOCIATE PROFESSOR",
            "A/Prof",
            "A/Prof.",
            "A / Prof",
            "A / Professor",
            "A / PROF",
            "Radiation Oncologist",
        ],
    )
    correspondence_recognizer = PatternRecognizer(
        supported_entity="CORRESPONDENCE",
        patterns=[
            Pattern(name="correspondence", regex=r"Dear(\s+)(\w+)(\s+)(\w+)", score=0.5)
        ],
    )  # default is 0.5, a lower score increases likelihood of capturing the entity but decreases the confidence
    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE",
        patterns=[
            Pattern(
                name="phone",
                regex=r"(\(+61\)|\+61|\(0[1-9]\)|0[1-9])?( ?-?[0-9]){8,14}",  # 8 to 14 digits
                score=0.5,
            )
        ],
    )
    mrn_recognizer = PatternRecognizer(
        supported_entity="MRN",
        patterns=[
            Pattern(
                name="mrn",
                regex=r"\d{5,9}",  # for numbers between 5-9 digits long
                score=0.5,
            )
        ],
    )
    providernumber_recognizer = PatternRecognizer(
        supported_entity="PROVIDER_NUMBER",
        patterns=[
            Pattern(
                name="provider number",
                regex=r"(\d+(Y))|(\d+(X)|(?:(Provider Number:)+\s+(\d+|\w+)))",
                score=0.5,
            )
        ],
    )
    date_recognizer = PatternRecognizer(
        supported_entity="DATE",
        patterns=[
            Pattern(
                name="date",
                regex=r"([0-9]{1,2}(\/|-|.)[0-9]{1,2}(\/|-|.)[0-9]{2,4})|(\b\d{1,2}\D{0,3})?\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?)\D?(\d{1,2}\D?)?\D?((19[7-9]\d|20\d{2})|\d{2})",
                score=0.5,
            )
        ],
    )
    street_recognizer = PatternRecognizer(
        supported_entity="STREET",
        patterns=[
            Pattern(
                name="street",
                regex=r"((\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE))|(\d+\s+\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce))|(\d+\s+\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE))|(\D+\S+\W+\S(?:ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE)(\s+\w+\s)(?:New South Wales|Victoria|Queensland|Western Australia|South Australia|Tasmania|Australian Capital Territory|Northern Territory|NEW SOUTH WALES|VICTORIA|QUEENSLAND|WESTERN AUSTRALIA|SOUTH AUSTRALIA|TASMANIA|AUSTRALIAN CAPITAL TERRITORY|NORTHERN TERRITORY|NSW|VIC|QLD|WA|SA|TAS|ACT|NT)(\s+\d{4})))",
                score=0.5,
            )
        ],
    )
    postcode_recognizer = PatternRecognizer(
        supported_entity="POSTCODE",
        patterns=[
            Pattern(
                name="postcode",
                regex=r"\d{4}",  # for numbers between 4 digits long
                score=0.5,
            )
        ],
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    suburbs_australia_path = os.path.join(script_dir, "suburbs_australia.txt")
    with open(suburbs_australia_path, "r", encoding='utf8') as f:
        deny_list = f.readlines()
    deny_list = [x.strip() for x in deny_list]
    suburb_recognizer = PatternRecognizer(
        supported_entity="SUBURB",
        deny_list=deny_list,
    )

    state_recognizer = PatternRecognizer(
        supported_entity="STATE",
        deny_list=[
            "NSW",
            "New South Wales",
            "NEW SOUTH WALES",
            "QLD",
            "Queensland",
            "QUEENSLAND",
            "NT",
            "Northern Territory",
            "NORTHERN TERRITORY",
            "WA",
            "Western Australia",
            "WESTERN AUSTRALIA",
            "SA",
            "South Australia",
            "SOUTH AUSTRALIA",
            "VIC",
            "Victoria",
            "VICTORIA",
            "TAS",
            "Tasmania",
            "TASMANIA",
            "ACT",
            "Australian Capital Territory",
            "AUSTRALIAN CAPITAL TERRITORY",
            "Australia",
            "AUSTRALIA",
        ],
    )

    institute_recognizer = PatternRecognizer(
        supported_entity="INSTITUTE",
        patterns=[
            Pattern(
                name="institute",
                regex=r"(\w+\s(Medical Centre|Cancer Centre|Medical Practice))",
                score=0.5,
            )
        ],
        deny_list=[
            "Prince of Wales Hospital",
            "Prince of Wales",
            "Prince of Wales Private",
            "POW Private",
            "POWPH",
            "POWH",
            "Nelune Comprehensive Cancer Centre",
            "Bright Building",
            "Liverpool Hospital",
            "Liverpool",
            "Campbelltown Hospital",
            "Campbelltown",
            "Wollongong Hospital",
            "Wollongong",
            "Shoalhaven District Memorial Hospital",
            "Shoalhaven District Memorial",
            "Shoalhaven",
            "St George Hospital",
            "St George",
            "SGH",
            "Royal North Shore Hospital",
            "Royal North Shore",
            "RNSH",
            "Tamworth Hospital",
            "Tamworth",
            "TBH",
            "Calvary",
            "Calvary Mater",
            "Calvary Mater Newcastle",
            "Calvary Mater Newcastle Hospital",
            "Newcastle",
            "CMMN",
            "St Vincents Hospital",
            "St Vincents",
            "GenesisCare",
            "SVH",
            "Macquarie Univerisity",
            "Macquarie University Hospital",
            "Waratah Private Hospital",
            "Hurstville",
            "Mater Sydney",
            "Mater Hospital",
            "Albury Wodonga",
            "Albury",
        ],
    )

    analyzer.registry.add_recognizer(title_recognizer)
    analyzer.registry.add_recognizer(correspondence_recognizer)
    analyzer.registry.add_recognizer(phone_recognizer)
    analyzer.registry.add_recognizer(mrn_recognizer)
    analyzer.registry.add_recognizer(providernumber_recognizer)
    analyzer.registry.add_recognizer(date_recognizer)
    analyzer.registry.add_recognizer(street_recognizer)
    analyzer.registry.add_recognizer(postcode_recognizer)
    analyzer.registry.add_recognizer(suburb_recognizer)
    analyzer.registry.add_recognizer(state_recognizer)
    analyzer.registry.add_recognizer(institute_recognizer)
    return analyzer


def _build_transformers():
    multilingual_tokenizer = AutoTokenizer.from_pretrained(
        "Babelscape/wikineural-multilingual-ner"
    )
    multilingual_model = AutoModelForTokenClassification.from_pretrained(
        "Babelscape/wikineural-multilingual-ner"
    )
    multilingual_nlp = pipeline(
        "ner",
        model=multilingual_model,
        tokenizer=multilingual_tokenizer,
        grouped_entities=True,
    )

    profession_tokenizer = AutoTokenizer.from_pretrained("BSC-NLP4BIA/prof-ner-cat-v1")
    profession_model = AutoModelForTokenClassification.from_pretrained(
        "BSC-NLP4BIA/prof-ner-cat-v1"
    )
    profession_nlp = pipeline(
        "ner",
        model=profession_model,
        tokenizer=profession_tokenizer,
        grouped_entities=True,
    )
    return multilingual_nlp, profession_nlp


def _anonymise_with_transformer(pipe, text) -> str:
    ner_results = pipe(text)
    for ner_result in ner_results:
        if ner_result["entity_group"] not in ["PER", "LOC", "ORG"]:
            continue
        text = text.replace(
            ner_result["word"], "[XXXX]"
        )  # if ner_result['score'] > 0.5 else text
    return text


def anonymise_image(ds, score_threshold=0.5):
    engine = DicomImageRedactorEngine()
    # ds = engine.redact(ds, fill="contrast")  # fill="background")

    analyser = _build_analyser()
    anonymizer = AnonymizerEngine()
    # operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[XXXX]"})}
    multilingual_nlp, profession_nlp = _build_transformers()
    for element in ds.elements():
        elem = ds[element.tag]
        if elem.VR == "PN":
            elem.value = ["XXXX"]
        elif elem.VR in [
            "LO",
            "LT",
            "OW",
            "SH",
            "ST",
            "UC",
            "UT",
        ]:  # https://dicom.nema.org/medical/dicom/current/output/html/part05.html#table_6.2-1
            try:
                analyzer_results = analyser.analyze(
                    text=elem.value, language="en", score_threshold=score_threshold
                )
                anonymized_text = anonymizer.anonymize(
                    text=elem.value,
                    analyzer_results=analyzer_results,
                    operators={
                        "DEFAULT": OperatorConfig("replace", {"new_value": "[XXXX]"})
                    },
                ).text
                anonymized_text = _anonymise_with_transformer(
                    multilingual_nlp, anonymized_text
                )
                anonymized_text = _anonymise_with_transformer(
                    profession_nlp, anonymized_text
                )
                elem.value = anonymized_text
            except:
                print(elem.tag)
    return ds


"""
if __name__ == "__main__":
    PROJECT_FOLDER = os.getcwd()
    IMG_PATH = os.path.join(PROJECT_FOLDER, 'sample_data/3_ORIGINAL.dcm')
    ds = dicom.dcmread(IMG_PATH)

    dicom_metadata = {str(elem.tag): _parse_value(elem.value) for elem in ds.elements()}
    df = pd.DataFrame.from_dict({k:[v] for k,v in dicom_metadata.items()})
    df.to_csv(IMG_PATH.replace(".dcm", ".csv"), index=False)

    redacted_dicom_instance = anonymise_image(ds)
    redacted_dicom_instance.save_as(IMG_PATH.replace('ORIGINAL', 'ANONYMIZED'))
"""
