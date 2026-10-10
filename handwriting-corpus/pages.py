"""
Source content for the synthetic handwriting corpus. Every string here must use
only characters the pretrained synthesis checkpoints were trained on:
    space ! " # % ' ( ) + , - . / 0-9 : ; ? A-Z [ a-z
(no '=', '*', '^', '>', '<', Greek letters, or superscripts -- see
checkpoints/Epoch_46/meta.json "charset"). Exam-style content below is adapted
from the same themes as the original photographed exam pages (regression
stats, k-fold, sampling, serial codes) but re-worded to fit that charset.

Each entry is (id, text, ground_truth_checks) where ground_truth_checks is a
list of substrings/tokens that must survive transcription for the page to
"pass" that check. strike_word, if present, names a token that will be struck
through in post-processing (degrade.py); correct_word is what should remain
legible nearby as the kept value.
"""

PAGES = [
    dict(
        id="p01_r2",
        text="The model reached an in sample R2 of point eight nine on the "
             "training set. Relationship to R2 on held out data was weaker.",
        checks=["in", "sample", "R2", "point", "eight", "nine", "training",
                "Relationship", "held", "out"],
    ),
    dict(
        id="p02_strike",
        text="Best model had the lowest R3 score among all candidates "
             "tested. Second best model kept the highest R2 instead.",
        checks=["lowest", "highest", "R2", "candidates", "Second", "best"],
        strike_word="R3",
        correct_word_line="highest R2",
    ),
    dict(
        id="p03_split",
        text="Data was split eighty twenty between training and test "
             "sets. This ratio was chosen after several trial runs.",
        checks=["eighty", "twenty", "training", "test", "ratio", "trial"],
    ),
    dict(
        id="p04_kfold",
        text="Cross validation used k minus one folds for training and "
             "one fold held out. We tried k equal to five here.",
        checks=["Cross", "validation", "k", "minus", "one", "folds",
                "five"],
    ),
    dict(
        id="p05_obs",
        text="Nine hundred observations were collected in total. Ten "
             "fold validation was applied across the whole set.",
        checks=["Nine", "hundred", "observations", "Ten", "fold",
                "validation"],
    ),
    dict(
        id="p06_range",
        text="Accuracy ranged from sixty percent to eighty percent "
             "across the folds. Variance was higher on smaller folds.",
        checks=["Accuracy", "sixty", "percent", "eighty", "Variance",
                "folds"],
    ),
    dict(
        id="p07_serial",
        text="Sample serial code QX 77419 BD2 was recorded on row "
             "twelve of the log sheet for this run.",
        checks=["QX", "77419", "BD2", "row", "twelve", "log"],
    ),
    dict(
        id="p08_labels",
        text="Label a marks the control group and label b marks the "
             "treatment group in this part of the study.",
        checks=["Label", "control", "group", "treatment", "study"],
    ),
    dict(
        id="p09_count",
        text="Count times n equals seven hundred twenty in the pooled "
             "sample across all three sites combined.",
        checks=["Count", "seven", "hundred", "twenty", "pooled",
                "three", "sites"],
    ),
    dict(
        id="p10_typo",
        text="Total obsevations after cleaning dropped to eight "
             "hundred and fifty from the original nine hundred.",
        checks=["obsevations", "eight", "hundred", "fifty", "nine"],
    ),
    dict(
        id="p11_strike2",
        text="Draft result showed a failing grade for this batch. "
             "Final result after recount showed a passing grade.",
        checks=["Draft", "Final", "recount", "passing", "grade"],
        strike_word="failing",
        correct_word_line="passing grade",
    ),
    dict(
        id="p12_dates",
        text="The pilot ran for six weeks starting the first week of "
             "the term and ending just before final exams began.",
        checks=["pilot", "six", "weeks", "term", "final", "exams"],
    ),
    dict(
        id="p13_ratio",
        text="Roughly one in four samples failed the quality check on "
             "the second pass through the pipeline.",
        checks=["Roughly", "one", "four", "failed", "quality", "second",
                "pipeline"],
    ),
    dict(
        id="p14_negnum",
        text="The residual mean was close to zero point one, well "
             "within the tolerance band set for this experiment.",
        checks=["residual", "mean", "zero", "point", "one", "tolerance"],
    ),
]
