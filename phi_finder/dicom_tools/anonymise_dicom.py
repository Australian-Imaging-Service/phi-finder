import os
from typing import Tuple

import pydicom as dicom

from presidio_image_redactor import DicomImageRedactorEngine
from presidio_anonymizer import AnonymizerEngine
from pydicom.pixel_data_handlers.util import apply_voi_lut
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline


def destroy_pixels(ds: dicom.dataset.FileDataset) -> dicom.dataset.FileDataset:
    """It sets all pixel values to 0.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset
        The DICOM dataset containing the image data to be destroyed.

    Returns
    -------
    pydicom.dataset.FileDataset
        The modified DICOM dataset with pixel data destroyed.
    """
    if "PixelData" in ds:
        pixel_array = ds.pixel_array
        pixel_array[:] = 0
        data_downsampling = pixel_array[:8, :8]
        ds.PixelData = data_downsampling.tobytes()
        ds.Rows, ds.Columns = data_downsampling.shape
    return ds


def _build_presidio_analyser(score_threshold: float=0.5,
                             spacy_model_name: str="en_core_web_md") -> AnalyzerEngine:
    """Builds and configures a Presidio analyser engine for named entity recognition.

    This function initialises an NLP engine using the SpaCy library and sets up
    various pattern recognisers for different types of entities, including titles,
    correspondence, phone numbers, medical record numbers (MRN), provider numbers,
    dates, street addresses, postcodes, suburbs, states, and institutes. The
    recognisers are configured with specific patterns and deny lists.

    Parameters
    ----------
    score_threshold : float, optional
        The score threshold for entity recognition. Entities with a score below this
        threshold will not be considered for anonymisation. Default is 0.5.
    spacy_model_name : str, optional
        The name of the SpaCy model to use for NLP processing. Default is "en_core_web_md".
        Other options include "en_core_web_sm" and "en_core_web_lg".
        
    Returns
    -------
    AnalyzerEngine
        An instance of the AnalyzerEngine configured with various recognisers for
        named entity recognition.
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": spacy_model_name}],
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
            Pattern(name="correspondence", regex=r"Dear(\s+)(\w+)(\s+)(\w+)", score=score_threshold)
        ],
    )  # A lower score increases likelihood of capturing the entity but decreases the confidence
    phone_recognizer = PatternRecognizer(
        supported_entity="PHONE",
        patterns=[
            Pattern(
                name="phone",
                regex=r"(\(+61\)|\+61|\(0[1-9]\)|0[1-9])?( ?-?[0-9]){8,14}",  # 8 to 14 digits
                score=score_threshold,
            )
        ],
    )
    mrn_recognizer = PatternRecognizer(
        supported_entity="MRN",
        patterns=[
            Pattern(
                name="mrn",
                regex=r"\d{5,9}",  # for numbers between 5-9 digits long
                score=score_threshold,
            )
        ],
    )
    gender_recognizer = PatternRecognizer(
        supported_entity="GENDER",
        patterns=[
            Pattern(
                name="gender",
                regex=r"(^[FfMm]$)|(^(?i)(male|female)$)",  # Sole string 'M' or 'F'
                score=score_threshold,
            )
        ],
    )
    providernumber_recognizer = PatternRecognizer(
        supported_entity="PROVIDER_NUMBER",
        patterns=[
            Pattern(
                name="provider number",
                regex=r"(\d+(Y))|(\d+(X)|(?:(Provider Number:)+\s+(\d+|\w+)))",
                score=score_threshold,
            )
        ],
    )
    date_recognizer = PatternRecognizer(
        supported_entity="DATE",
        patterns=[
            Pattern(
                name="date",
                regex=r"([0-9]{1,2}(\/|-|.)[0-9]{1,2}(\/|-|.)[0-9]{2,4})|(\b\d{1,2}\D{0,3})?\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?)\D?(\d{1,2}\D?)?\D?((19[7-9]\d|20\d{2})|\d{2})",
                score=score_threshold,
            )
        ],
    )
    street_recognizer = PatternRecognizer(
        supported_entity="STREET",
        patterns=[
            Pattern(
                name="street",
                regex=r"((\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE))|(\d+\s+\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce))|(\d+\s+\w+\s(?:Alley|Ally|Arcade|Arc|Avenue|Ave|Boulevard|Bvd|Bypass|Bypa|Circuit|CCt|Close|Corner|Crn|Court|Crescent|Cres|Cul-de-sac|Cds|Drive|Esplanade|Esp|Green|Grn|Grove|Highway|Hwy|Junction|Jnc|Lane|Link|Mews|Parade|Pde|Place|Ridge|Rdge|Road|Rd|Square|Street|Terrace|Tce|ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE))|(\D+\S+\W+\S(?:ALLEY|ALLY|ARCADE|ARC|AVENUE|AVE|BOULEVARD|BVD|BYPASS|BYPA|CIRCUIT|CCT|CLOSE|CORNER|CRN|COURT|CRESCENT|CRES|CUL-DE-SAC|CDS|DRIVE|ESPLANADE|ESP|GREEN|GRN|GROVE|HIGHWAY|HWY|JUNCTION|JNC|LANE|LINK|MEWS|PARADE|PDE|PLACE|RIDGE|RDGE|ROAD|RD|SQUARE|STREET|TERRACE|TCE)(\s+\w+\s)(?:New South Wales|Victoria|Queensland|Western Australia|South Australia|Tasmania|Australian Capital Territory|Northern Territory|NEW SOUTH WALES|VICTORIA|QUEENSLAND|WESTERN AUSTRALIA|SOUTH AUSTRALIA|TASMANIA|AUSTRALIAN CAPITAL TERRITORY|NORTHERN TERRITORY|NSW|VIC|QLD|WA|SA|TAS|ACT|NT)(\s+\d{4})))",
                score=score_threshold,
            )
        ],
    )
    postcode_recognizer = PatternRecognizer(
        supported_entity="POSTCODE",
        patterns=[
            Pattern(
                name="postcode",
                regex=r"\d{4}",  # for numbers between 4 digits long
                score=score_threshold,
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
                score=score_threshold,
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
    analyzer.registry.add_recognizer(gender_recognizer)
    analyzer.registry.add_recognizer(date_recognizer)
    analyzer.registry.add_recognizer(street_recognizer)
    analyzer.registry.add_recognizer(postcode_recognizer)
    analyzer.registry.add_recognizer(suburb_recognizer)
    analyzer.registry.add_recognizer(state_recognizer)
    analyzer.registry.add_recognizer(institute_recognizer)
    return analyzer


def _build_transformers() -> Tuple[TokenClassificationPipeline, TokenClassificationPipeline]:
    """Builds and returns named entity recognition (NER) pipelines for multilingual and profession-specific models.

    This function initializes two NER pipelines:
    1. A multilingual NER pipeline using the "Babelscape/wikineural-multilingual-ner" model.
    2. A profession-specific NER pipeline using the "BSC-NLP4BIA/prof-ner-cat-v1" model.

    Returns
    -------
    tuple
        A tuple containing two Huggingface's TokenClassificationPipeline:
        - multilingual_nlp: The multilingual NER pipeline.
        - profession_nlp: The profession-specific NER pipeline.
    """
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


def _anonymise_with_transformer(pipe: TokenClassificationPipeline, text: str) -> str:
    """Anonymises text using a specified named entity recognition (NER) pipeline.

    This function processes the input text through the provided NER pipeline,
    replacing recognised entities of type "PER", "LOC", and "ORG" with the placeholder "[XXXX]".

    Parameters
    ----------
    pipe : Huggingface's TokenClassificationPipeline
        The NER pipeline to use for entity recognition.
    
    text : str
        The input text to be anonymised.

    Returns
    -------
    str
        The anonymised text with specified entities replaced by "[XXXX]".
    """
    ner_results = pipe(text)
    for ner_result in ner_results:
        if ner_result["entity_group"] not in ["PER", "LOC", "ORG"]:
            continue
        text = text.replace(
            ner_result["word"], "[XXXX]"
        )  # if ner_result['score'] > 0.5 else text
    return text


def anonymise_image(ds: dicom.dataset.FileDataset,
                    analyser: AnalyzerEngine=None,
                    anonymizer: AnonymizerEngine=None,
                    score_threshold: float=0.5,
                    use_transformers: bool=False) -> dicom.dataset.FileDataset:
    """Anonymises a DICOM image by redacting personal information.

    This function processes the DICOM dataset, redacting personal names and other
    identifiable information based on the specified score threshold. It utilises
    named entity recognition pipelines to identify and replace sensitive information.

    Parameters
    ----------
    ds : pydicom.dataset.FileDataset
        The DICOM dataset containing the image data and metadata to be anonymised.
    
    score_threshold : float, optional
        The score threshold for entity recognition. Entities with a score below this
        threshold will not be considered for anonymisation. Default is 0.5.

    use_transformers : bool, optional (default False)
        If True, transformers will be used for anonymisation on top of Presidio's output.

    Returns
    -------
    pydicom.dataset.FileDataset
        The anonymised DICOM.
    """
    #engine = DicomImageRedactorEngine()
    # ds = engine.redact(ds, fill="contrast")  # fill="background")
    if analyser is None:
        analyser = _build_presidio_analyser(score_threshold)
    if anonymizer is None:
        anonymizer = AnonymizerEngine()
    # operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[XXXX]"})}
    if use_transformers:
        multilingual_nlp, profession_nlp = _build_transformers()
    for element in ds.elements():
        elem = ds[element.tag]
        if elem.VR == "PN":
            elem.value = ["XXXX"]
        elif elem.VR in [
            "LO",  # Long String
            "LT",  # Long Text
            "OW",  # Other Word
            "SH",  # Short String
            "ST",  # Short Text
            "UC",  # Unlimited Characters
            "UT",  # Unlimited Text
            "DA",  # Date
            "CS",  # Code String
            "AS",  # Age String
        ]:  # https://dicom.nema.org/medical/dicom/current/output/html/part05.html#table_6.2-1 and https://pydicom.github.io/pydicom/stable/guides/element_value_types.html
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
                if use_transformers:
                    anonymized_text = _anonymise_with_transformer(
                        multilingual_nlp, anonymized_text
                    )
                    anonymized_text = _anonymise_with_transformer(
                        profession_nlp, anonymized_text
                    )
                elem.value = anonymized_text
            except:
                print(elem.tag)  # pixel data falls here.
    return ds
