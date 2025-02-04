import argparse
import os
from llms import get_llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="query llm. ",
    )

    parser.add_argument(
        "--question_dir",
        default="./eval_result/questions.txt",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="${OPENAI_API_KEY}"
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

    if args.question_dir:
        with open(args.question_dir, "r") as f:
            questions = [question.rstrip("\n") for question in f]
    else:
        questions = ["How do amyloid beta circadian patterns change with respect to age and amyloid pathology"]


    system_prompt = """---Role---

You are a helpful assistant responding to questions.


---Goal---

Generate a response of the target length and format that responds to the user's question leverage any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.

---Target response length and format---

Multiple Paragraphs


Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown."""

    results = llm.inference(
        system_prompt=system_prompt,
        user_contents=questions)

    llm.save_results(filename=f"{args.output_dir}/{args.model_name}_answer.json", results=results)


    '''
    LightRAG system prompt
    """
    
    ---Role---
    
    You are a helpful assistant responding to questions.
    
    
    ---Goal---
    
    Generate a response of the target length and format that responds to the user's question leverage any relevant general knowledge.
    If you don't know the answer, just say so. Do not make anything up.
    
    ---Target response length and format---
    
    Multiple Paragraphs
    
    
    Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown."""
    
    '''

    '''
    
    '''