from langchain_community.document_loaders import PyMuPDFLoader
# from llama_index.readers.file import PyMuPDFReader
import os
import os.path as osp
import argparse


def convert_pdf(args):
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)
    for file in os.listdir(args.input_dir):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(osp.join(args.input_dir, file))
            data = loader.load()
            temp = ""
            for i in data:
                temp += i.page_content
            with open(osp.join(args.output_dir, f"{file}.txt"), "w") as f:
                f.write(temp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pdf converter",
    )
    parser.add_argument(
        "--input_dir",
        default="./raw_paper_pdf_0122",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        default="./raw_paper_txt",
        type=str
    )
    args = parser.parse_args()
    convert_pdf(args)


        
    