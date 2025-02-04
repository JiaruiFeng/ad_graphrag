import anthropic
import time
from typing import List, Dict
import asyncio
from datetime import datetime
import json
import os
import argparse
from eval_prompts import *
from llms import get_llm
import re

prompt_dict = {
    "comprehensive": COMPREHENSIVENESS_PROMPT,
    "diversity": DIVERSITY_PROMPT,
    "empowerment": EMPOWERMENT_PROMPT,
    "directness": DIRECTNESS_PROMPT,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="evaluate graph rag. ",
    )

    parser.add_argument(
        "--direction",
        help="evaluation direction",
        required=False,
        default="comprehensive",
        choices=("comprehensive", "diversity", "empowerment", "directness"),
    )
    parser.add_argument(
        "--question_dir",
        default="./eval_result/questions.txt",
        type=str,
    )
    parser.add_argument(
        "--rag_ans_dir",
        default="./eval_result/lightrag_hybrid_new_prompt_result.json",
        type=str,
    )
    parser.add_argument(
        "--llm_ans_dir",
        default="./eval_result/gpt-4o_answer.json",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="claude-3-5-sonnet-20241022"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="${CLAUDE_API_KEY}"
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default=None
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=None
    )
    parser.add_argument(
        "--organization",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_result",
    )

    args = parser.parse_args()
    args.api_key = os.path.expandvars(args.api_key)

    llm = get_llm(args)

    query_prompt = prompt_dict[args.direction]
    if args.question_dir:
        with open(args.question_dir, "r") as f:
            questions = [question.rstrip("\n") for question in f]
    else:
        questions = ["How do amyloid beta circadian patterns change with respect to age and amyloid pathology"]

    if args.rag_ans_dir:
        with open(args.rag_ans_dir, "r") as f:
            results = json.load(f)
        rag_ans = [result["response"] for result in results]
    else:
        rag_ans = ["""### Overview of Amyloid Beta Removal

Amyloid-beta (Aβ) is a peptide that accumulates in the brain and is a hallmark of Alzheimer's disease. The removal of amyloid-beta from the brain involves complex processes of synthesis and clearance, which are critical in understanding and potentially treating Alzheimer's disease.

### Mechanisms of Removal

1. **Synthesis and Clearance Rates**: The removal of amyloid-beta is influenced by its synthesis and clearance rates. Researchers utilize techniques such as stable isotope labeling tandem mass spectrometry to quantify these rates, providing insights into the dynamics of amyloid-beta in the brain [Data: Reports (132)].

2. **Role of APOE Gene**: The APOE gene plays a significant role in lipid metabolism and impacts the deposition and clearance of amyloid-beta in the central nervous system. Variations in this gene can moderate amyloid-beta-related cognitive decline, highlighting its importance in the pathology of Alzheimer's disease [Data: Reports (33)].

3. **Gamma-Secretase Inhibitors**: These are a class of drugs that decrease amyloid-beta production in the central nervous system. By inhibiting gamma-secretase, these drugs potentially reduce the accumulation of amyloid-beta, offering a therapeutic approach to Alzheimer's disease [Data: Reports (132)].

### Therapeutic Approaches

1. **Monoclonal Antibodies**: Anti-amyloid monoclonal antibodies, such as Gantenerumab, Bapineuzumab, and Aducanumab, target amyloid-beta plaques in the brain. These therapies aim to reduce amyloid-beta levels and are being evaluated in various clinical trials for their efficacy and safety. However, they may lead to Amyloid-Related Imaging Abnormalities (ARIA), which are significant adverse events [Data: Reports (4, 69, 78, 79, 126)].

2. **Clinical Trials**: Several clinical trials, including the GRADUATE phase 3 trials, SCARLET ROAD, and others, are investigating the efficacy of drugs like Gantenerumab in reducing amyloid-beta plaques and cognitive decline in Alzheimer's patients. These trials are crucial in determining the potential of these drugs as treatment options [Data: Reports (69, 124)].

### Conclusion

The removal of amyloid-beta from the brain is a multifaceted process involving genetic, biochemical, and therapeutic factors. Understanding these mechanisms is essential for developing effective treatments for Alzheimer's disease. Ongoing research and clinical trials continue to explore these avenues, offering hope for future therapeutic interventions.
"""]

    if args.llm_ans_dir:
        with open(args.llm_ans_dir, "r") as f:
            results = json.load(f)
        llm_ans = [result["response"] for result in results]

    else:
        llm_ans = ["""Amyloid-beta (Aβ) circadian patterns, which refer to the daily fluctuations in Aβ levels in the brain, are known to change with age and the development of amyloid pathology. Here’s a summary of how these patterns are affected:
1. Circadian Rhythms in Young vs. Old Individuals:
Young Individuals: In healthy young individuals, Aβ levels exhibit a clear circadian rhythm. Aβ concentrations are typically lower during sleep and higher during wakefulness. This rhythmic fluctuation is believed to be regulated by sleep-wake cycles and the glymphatic system, which is more active during sleep and facilitates the clearance of Aβ from the brain.
Aging: As individuals age, the amplitude of these circadian fluctuations in Aβ decreases. This means that the difference between the peak and trough Aβ levels throughout the day becomes less pronounced. The circadian regulation of Aβ becomes less efficient, contributing to higher overall Aβ levels.
2. Impact of Amyloid Pathology:
Early Stages of Amyloid Pathology: In the early stages of amyloid pathology, circadian rhythms of Aβ start to become disrupted. The clearance of Aβ becomes less effective, leading to a gradual accumulation of Aβ, particularly during periods when levels should typically be lower (e.g., during sleep).
Advanced Amyloid Pathology (Alzheimer’s Disease): In individuals with advanced amyloid pathology, such as in Alzheimer’s disease (AD), the circadian rhythm of Aβ is often severely disrupted or even completely lost. Aβ levels may remain consistently high throughout the day, which is thought to contribute to the aggregation of Aβ into plaques. This disruption is also associated with sleep disturbances commonly observed in AD patients.
3. Consequences of Disrupted Aβ Circadian Rhythms:
The loss of circadian regulation of Aβ is both a contributor to and a consequence of amyloid pathology. Disrupted sleep patterns, common in aging and AD, further impair the clearance of Aβ, creating a vicious cycle that accelerates amyloid deposition and neurodegeneration.
Studies suggest that interventions aimed at restoring healthy circadian rhythms, such as improving sleep hygiene or using light therapy, might help mitigate the accumulation of Aβ and slow the progression of amyloid pathology.
Summary
As individuals age and develop amyloid pathology, the circadian rhythm of Aβ becomes increasingly disrupted, leading to higher and more constant levels of Aβ. This disruption is linked to the progression of Alzheimer’s disease and other neurodegenerative conditions. Maintaining healthy sleep and circadian rhythms could be a potential therapeutic strategy to manage Aβ levels and reduce the risk of amyloid-related diseases.

"""]

    queries = []
    for q, rag_a, llm_a in zip(questions, rag_ans, llm_ans):
        queries.append(query_prompt.format(QUESTION=q, GRAPH_RAG_ANSWER=rag_a, CHAT_LLM_ANSWER=llm_a))


    system_prompt = """You are a helpful AI assistant. Please provide concise, 
    accurate responses to queries. If you're unsure about something, say so."""

    results = llm.inference(
        system_prompt=system_prompt,
        user_contents=queries)

    prefix = args.rag_ans_dir.split("/")[-1].split(".")[0]
    llm.save_results(filename=f"{args.output_dir}/eval_{prefix}_{args.direction}_result.json", results=results)

    better = 0
    total = 0
    for result in results:
        if result['status'] == "success":
            pattern = r"<choice>\s*(.*?)\s*</choice>"
            answer = re.findall(pattern, result["response"])[0]
            if answer.lower() == "graph rag":
                better += 1
            total += 1
    print(f"Evaluation on {args.direction}")
    print("Total questions: ", total)
    print("RAG win: ", better)
    print("Win rate: ", better / total)