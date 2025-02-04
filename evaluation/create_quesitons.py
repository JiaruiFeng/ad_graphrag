#
# QUESTION_LIST = [
#     """
#     Please list the various amyloid beta species that have been measured by stable isotope labeling kinetics (SILK). Tell me about their turnover kinetics relative to each-other. How do these kinetics vary in people who are amyloid-positive. How is amyloid-positivity defined in these kinetic studies?
#     """,
#     """
#     What can the plasma 42:40 biomarker tell you about your risk for becoming PET positive in the future?
#     """,
#     """
#     Please describe the various forms of APOE, it’s relevance to Alzheimer's disease and what amino acid changes occur in each variant. How do these amino acid changes affect the function of APOE?
#     """,
#     """
#     Please define late onset Alzheimer’s disease. What age does it occur at? How does amyloid beta production and clearance affect late onset Alzheimer's disease progression and risk? Please elaborate with evidence supporting your answer.
#     """,
#     """
#     What is are the biggest risk factors for late onset Alzheimer’s disease? What is known about how these risk factors affect the kinetics of amyloid beta isoforms and what evidence supports this? Please elaborate and hypothesize on the mechanisms behind this.
#     """,
#     """
#     Define tauopathy. What is known about tau protein, phosphorylation of tau and its effect on Alzheimer's disease. How do abnormal tau phosphorylation patterns affect tauopathy and Alzheimer's disease? What are the hypothesized mechanisms behind this and what evidence do we have to support these hypothesizes?
#     """,
#     """
#     Does exercise have an effect on Alzheimer's disease? What biomarkers or measurements were used to determine this? If exercise slows or accelerates the progression of Alzheimer’s disease, please hypothesize why that might be.
#     """,
#     """
#     How is amyloid beta removed from the brain?
#     """,
#     """
#     How do amyloid beta circadian patterns change with respect to age and amyloid pathology?
#     """
# ]
import pandas as pd

FILE_DIR = "evaluation/eval_result"

if __name__ == "__main__":
    question_list = pd.read_csv(f"{FILE_DIR}/question_list.csv")
    question_list = question_list["Question"].tolist()

    with open(f"{FILE_DIR}/questions.txt", "w") as f:
        for question in question_list:
            f.write(f"{question.lstrip().rstrip()}\n")
    # test
    with open(f"{FILE_DIR}/questions.txt", "r") as f:
        questions = [question.rstrip("\n") for question in f]
