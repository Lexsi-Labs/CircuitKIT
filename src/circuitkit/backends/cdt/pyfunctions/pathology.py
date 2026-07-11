import copy
import random

"""
Utility functions for the pathology report dataset we use as an example to demonstrate CD-T on BERTs.
"""

kidney_fields = {
    "classification": [
        "HistologicType_H",
        "LymphovascularInvasion_L",
        "Margins_Ma",
        "Procedure_Pr",
        "SpecimenLaterality_S",
        "TumorExtension_TumorE",
        "TumorSite_T",
    ],
    "token_extraction": [
        "MRN_M0",
        "AccessionNumber_A0",
        "NumberOfLymphNodesExamined_Nu0",
        "TumorSizeGreatestDimension_Tum0",
        "NumberOfLymphNodesInvolved_N0",
        "pT_p0",
        "pN_pN0",
        "pM_pM0",
    ],
    "transfer": ["HistologicGrade_Hi", "LymphovascularInvasion_L"],
    "string_similarity": ["TumorSite_T0", "Procedure_Pr0", "HistologicType_H0"],
}

lung_fields = {
    "classification": [
        "HistologicGrade_Hi",
        "LymphovascularInvasion_L",
        "PerineuralInvasion_Pe",
        "Procedure_Pr",
        "SpecimenLaterality_S",
        "TumorFocality_Tum",
        "TumorSite_T",
    ],
    "token_extraction": [
        "MRN_M0",
        "AccessionNumber_A0",
        "NumberOfLymphNodesExamined_Num0",
        "TumorSizeGreatestDimension_Tu0",
        "NumberOfLymphNodesInvolved_N0",
        "pT_p0",
        "pN_pN0",
        "pM_pM0",
    ],
    "transfer": ["HistologicGrade_Hi", "LymphovascularInvasion_L"],
    "string_similarity": ["TumorSite_T0", "Procedure_Pr0", "HistologicType_H0"],
}

colon_fields = {
    "classification": [
        "HistologicGrade_Hi",
        "HistologicType_H",
        "LymphovascularInvasion_L",
        "PerineuralInvasion_Pe",
        "Procedure_Pr",
        "TumorSite_T",
    ],
    "token_extraction": [
        "MRN_M0",
        "AccessionNumber_A0",
        "NumberOfLymphNodesExamined_Nu0",
        "TumorSizeGreatestDimension_Tum0",
        "NumberOfLymphNodesInvolved_N0",
        "pT_p0",
        "pN_pN0",
        "pM_pM0",
    ],
    "transfer": ["HistologicGrade_Hi", "LymphovascularInvasion_L"],
    "string_similarity": ["HistologicType_H0", "TumorSite_T0"],
}

prostate_fields = {
    "classification": [
        "TreatmentEffect",
        "TumorType",
        "PrimaryGleason",
        "SecondaryGleason",
        "TertiaryGleason",
        "SeminalVesicleNone",
        "LymphNodesNone",
        "MarginStatusNone",
        "ExtraprostaticExtension",
        "PerineuralInfiltration",
        "RbCribriform",
        "BenignMargins",
    ],
    "token_extraction": ["ProstateWeight", "TumorVolume", "TStage", "NStage", "MStage"],
}


def fixLabel(data):
    for split_ in ["train", "val", "test", "dev_test"]:
        for i in range(len(data[split_])):
            for field in ["PrimaryGleason", "SecondaryGleason", "TertiaryGleason"]:
                if data[split_][i]["labels"][field] == 0 or data[split_][i]["labels"][field] == "0":
                    data[split_][i]["labels"][field] = "null"
                elif data[split_][i]["labels"][field] == 2:
                    data[split_][i]["labels"][field] = "2"
                elif data[split_][i]["labels"][field] == 3:
                    data[split_][i]["labels"][field] = "3"
                elif data[split_][i]["labels"][field] == 4:
                    data[split_][i]["labels"][field] = "4"
                elif data[split_][i]["labels"][field] == 5:
                    data[split_][i]["labels"][field] = "5"

            for field in ["MarginStatusNone", "SeminalVesicleNone"]:
                if data[split_][i]["labels"][field] == 0:
                    data[split_][i]["labels"][field] = "0"  # positive
                elif data[split_][i]["labels"][field] == 1:
                    data[split_][i]["labels"][field] = "1"  # negative
    return data


# exclude '2' and 'null'
def exclude_labels(docs, labels):
    ex, final_d, final_l = [], [], []
    for i, l in enumerate(labels):
        if l == "2" or l == "null":
            ex.append(i)
    for i, (d, l) in enumerate(zip(docs, labels)):
        if i not in ex:
            final_d.append(d)
            final_l.append(l)
    return final_d, final_l


def fixLabelProstateGleason(data):
    for split_ in ["train", "val", "test", "dev_test"]:
        for i in range(len(data[split_])):
            for field in ["PrimaryGleason", "SecondaryGleason", "TertiaryGleason"]:
                if data[split_][i]["labels"][field] == 0:
                    data[split_][i]["labels"][field] = "null"
    return data


def extract_synoptic(text, tokenizer):

    phrases = [
        "synoptic comment",
        "final pahtologic diagnosis",
        "diagnosis comment",
        "clinical diagnosis",
        "diagnosis :",
    ]
    for phrase in phrases:
        if phrase in text:
            ind = text.index(phrase)
            text = text[ind:]
            tokens = tokenizer.encode(text)
            n = len(tokens)
            tokens = tokens[1 : min(n - 1, 513)]

            return tokenizer.decode(tokens)
    return text


def reinitializeSplit(data, sizes, i):
    random.seed(i)
    full = copy.deepcopy(data["train"] + data["val"] + data["test"])
    random.shuffle(full)
    data["train"] = full[: sizes["train"]]
    data["val"] = full[sizes["train"] : sizes["val"] + sizes["train"]]
    data["test"] = full[sizes["val"] + sizes["train"] :]
    return data


def getProstateMapping():
    """
    * Return dic mapping of original raw labels back to processed labels
    """
    mapping = {
        "null": 0,
        "": 0,
        "0": 1,
        "o": 1,
        "1": 2,
        "1a": 3,
        "1b": 4,
        "1c": 5,
        "2": 6,
        "2a": 7,
        "2b": 8,
        "2c": 9,
        "*s": 10,
        "3": 11,
        "3a": 12,
        "3b": 13,
        "3c": 14,
        "4": 15,
        "4a": 16,
        "5": 17,
        "t2": 18,
        "t2a": 19,
        "x": 20,
        "adenoca": 21,
        "capsmac": 21,
        "capdctl": 21,
        "sarcoma": 22,
        "mucinous": 21,
        "uccinvas": 21,
    }
    return mapping


def getProstateStageInverseMapping():
    mapping = {
        0: "",
        "1": 2,
        "1a": 3,
        "1b": 4,
        "1c": 5,
        "2": 6,
        "2a": 7,
        "2b": 8,
        "2c": 9,
        "*s": 10,
        "3": 11,
        "3a": 12,
        "3b": 13,
        "3c": 14,
        "4": 15,
        "4a": 16,
        "5": 17,
        "t2": 18,
        "t2a": 19,
        "x": 20,
        "adeno": 21,
        "sarcoma": 22,
    }
    mapping = {v: k for k, v in mapping.items()}
    mapping["0"] = ""
    mapping[0] = ""
    mapping[1] = "0"

    return mapping


def fixProstateLabels(data):

    data = fixLabelProstateGleason(data)
    for split in ["train", "val", "test"]:

        for patient in data[split]:
            patient["clean_labels"] = patient["labels"].copy()

    data["train"][270]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][270]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][324]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][324]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][380]["clean_labels"]["PrimaryGleason"] = 3
    data["train"][380]["clean_labels"]["SecondaryGleason"] = 4

    data["train"][509]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][509]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][522]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][522]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][588]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][588]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][746]["clean_labels"]["PrimaryGleason"] = 3
    data["train"][746]["clean_labels"]["SecondaryGleason"] = 4

    data["train"][769]["clean_labels"]["PrimaryGleason"] = 4
    data["train"][769]["clean_labels"]["SecondaryGleason"] = 3

    data["train"][773]["clean_labels"]["PrimaryGleason"] = 3
    data["train"][773]["clean_labels"]["SecondaryGleason"] = 4

    data["train"][786]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][786]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][830]["clean_labels"]["PrimaryGleason"] = 3
    data["train"][830]["clean_labels"]["SecondaryGleason"] = 3

    data["train"][838]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][838]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][945]["clean_labels"]["PrimaryGleason"] = 4
    data["train"][945]["clean_labels"]["SecondaryGleason"] = 3

    data["train"][1150]["clean_labels"]["PrimaryGleason"] = 4
    data["train"][1150]["clean_labels"]["SecondaryGleason"] = 3

    data["train"][1234]["clean_labels"]["PrimaryGleason"] = 3
    data["train"][1234]["clean_labels"]["SecondaryGleason"] = 4

    data["train"][1257]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][1257]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][1288]["clean_labels"]["PrimaryGleason"] = 5
    data["train"][1288]["clean_labels"]["SecondaryGleason"] = 4

    data["train"][1390]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][1390]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][1404]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][1404]["clean_labels"]["SecondaryGleason"] = "null"
    data["train"][1404]["clean_labels"]["TertiaryGleason"] = "null"
    data["train"][1404]["clean_labels"]["SeminalVesicleNone"] = 1

    data["train"][1407]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][1407]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][1439]["clean_labels"]["PrimaryGleason"] = 5
    data["train"][1439]["clean_labels"]["SecondaryGleason"] = 3

    data["train"][1484]["clean_labels"]["PrimaryGleason"] = 5
    data["train"][1484]["clean_labels"]["SecondaryGleason"] = 3

    data["train"][1491]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][1491]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][1711]["clean_labels"]["PrimaryGleason"] = 4
    data["train"][1711]["clean_labels"]["SecondaryGleason"] = 5

    data["train"][1724]["clean_labels"]["PrimaryGleason"] = 4
    data["train"][1724]["clean_labels"]["SecondaryGleason"] = 4

    data["train"][1793]["clean_labels"]["PrimaryGleason"] = 4
    data["train"][1793]["clean_labels"]["SecondaryGleason"] = 3

    data["train"][1831]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][1831]["clean_labels"]["SecondaryGleason"] = "null"

    data["train"][1919]["clean_labels"]["PrimaryGleason"] = "null"
    data["train"][1919]["clean_labels"]["SecondaryGleason"] = "null"

    data["val"][24]["clean_labels"]["PrimaryGleason"] = "null"
    data["val"][24]["clean_labels"]["SecondaryGleason"] = "null"

    data["val"][56]["clean_labels"]["PrimaryGleason"] = "null"
    data["val"][56]["clean_labels"]["SecondaryGleason"] = "null"

    data["val"][120]["clean_labels"]["PrimaryGleason"] = "null"
    data["val"][120]["clean_labels"]["SecondaryGleason"] = "null"
    data["val"][120]["clean_labels"]["TertiaryGleason"] = "null"
    data["val"][120]["clean_labels"]["SeminalVesicleNone"] = 1

    data["val"][122]["clean_labels"]["PrimaryGleason"] = 5
    data["val"][122]["clean_labels"]["SecondaryGleason"] = 4

    data["val"][185]["clean_labels"]["PrimaryGleason"] = 4
    data["val"][185]["clean_labels"]["SecondaryGleason"] = 5

    data["val"][240]["clean_labels"]["PrimaryGleason"] = 3
    data["val"][240]["clean_labels"]["SecondaryGleason"] = 3

    data["val"][258]["clean_labels"]["PrimaryGleason"] = 3
    data["val"][258]["clean_labels"]["SecondaryGleason"] = "null"

    data["val"][311]["clean_labels"]["PrimaryGleason"] = "null"
    data["val"][311]["clean_labels"]["SecondaryGleason"] = "null"
    data["val"][311]["clean_labels"]["TertiaryGleason"] = "null"

    data["val"][319]["clean_labels"]["PrimaryGleason"] = 4
    data["val"][319]["clean_labels"]["SecondaryGleason"] = 5

    data["val"][362]["clean_labels"]["PrimaryGleason"] = 3
    data["val"][362]["clean_labels"]["SecondaryGleason"] = 4

    data["val"][383]["clean_labels"]["PrimaryGleason"] = 4
    data["val"][383]["clean_labels"]["SecondaryGleason"] = 5

    data["val"][441]["clean_labels"]["PrimaryGleason"] = "null"
    data["val"][441]["clean_labels"]["SecondaryGleason"] = "null"

    data["test"][6]["clean_labels"]["PrimaryGleason"] = "null"
    data["test"][6]["clean_labels"]["SecondaryGleason"] = "null"

    data["test"][24]["clean_labels"]["PrimaryGleason"] = "null"
    data["test"][24]["clean_labels"]["SecondaryGleason"] = "null"

    data["test"][94]["clean_labels"]["PrimaryGleason"] = "null"
    data["test"][94]["clean_labels"]["SecondaryGleason"] = "null"

    data["test"][98]["clean_labels"]["PrimaryGleason"] = 3
    data["test"][98]["clean_labels"]["SecondaryGleason"] = "null"

    data["test"][100]["clean_labels"]["PrimaryGleason"] = 3
    data["test"][100]["clean_labels"]["SecondaryGleason"] = 4

    data["test"][101]["clean_labels"]["PrimaryGleason"] = 4
    data["test"][101]["clean_labels"]["SecondaryGleason"] = 3

    data["test"][118]["clean_labels"]["PrimaryGleason"] = "null"
    data["test"][118]["clean_labels"]["SecondaryGleason"] = "null"

    data["test"][120]["clean_labels"]["PrimaryGleason"] = "null"
    data["test"][120]["clean_labels"]["SecondaryGleason"] = "null"

    data["test"][132]["clean_labels"]["PrimaryGleason"] = 3
    data["test"][132]["clean_labels"]["SecondaryGleason"] = 3

    data["test"][319]["clean_labels"]["PrimaryGleason"] = 3
    data["test"][319]["clean_labels"]["SecondaryGleason"] = 4

    data["train"][47]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][119]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][495]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][616]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][704]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][773]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][795]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][873]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][996]["clean_labels"]["SeminalVesicleNone"] = 0
    data["train"][1096]["clean_labels"]["SeminalVesicleNone"] = 0

    data["val"][11]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][58]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][93]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][156]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][162]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][211]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][268]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][275]["clean_labels"]["SeminalVesicleNone"] = 0
    data["val"][311]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][350]["clean_labels"]["SeminalVesicleNone"] = 0
    data["val"][362]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][454]["clean_labels"]["SeminalVesicleNone"] = 0
    data["val"][455]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][478]["clean_labels"]["SeminalVesicleNone"] = 1
    data["val"][508]["clean_labels"]["SeminalVesicleNone"] = 1

    data["test"][28]["clean_labels"]["SeminalVesicleNone"] = 1
    data["test"][55]["clean_labels"]["SeminalVesicleNone"] = 0
    data["test"][91]["clean_labels"]["SeminalVesicleNone"] = 1
    data["test"][296]["clean_labels"]["SeminalVesicleNone"] = 0

    data["train"][471]["clean_labels"]["MarginStatusNone"] = 0
    data["train"][773]["clean_labels"]["MarginStatusNone"] = 0
    data["train"][1675]["clean_labels"]["MarginStatusNone"] = 0
    data["train"][1948]["clean_labels"]["MarginStatusNone"] = 0

    data["val"][56]["clean_labels"]["MarginStatusNone"] = 1
    data["val"][110]["clean_labels"]["MarginStatusNone"] = 0
    data["val"][128]["clean_labels"]["MarginStatusNone"] = 0
    data["val"][211]["clean_labels"]["MarginStatusNone"] = 0
    data["val"][257]["clean_labels"]["MarginStatusNone"] = 0
    data["val"][362]["clean_labels"]["MarginStatusNone"] = 1
    data["val"][384]["clean_labels"]["MarginStatusNone"] = 1
    data["val"][505]["clean_labels"]["MarginStatusNone"] = 1

    data["test"][6]["clean_labels"]["MarginStatusNone"] = 1
    data["test"][47]["clean_labels"]["MarginStatusNone"] = 1
    data["test"][81]["clean_labels"]["MarginStatusNone"] = 0
    data["test"][168]["clean_labels"]["MarginStatusNone"] = 1
    data["test"][228]["clean_labels"]["MarginStatusNone"] = 0
    data["test"][319]["clean_labels"]["MarginStatusNone"] = 1
    return data
