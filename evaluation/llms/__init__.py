from .claude import ChatClaude
from .oai import ChatOpenAI
def get_llm(args):
    if args.model_name.startswith("claude"):
        return ChatClaude(
            api_key=args.api_key,
            model=args.model_name,
            batch_size=args.batch_size)
    else:
        return ChatOpenAI(
            api_key=args.api_key,
            api_base=args.api_base,
            api_version=args.api_version,
            organization=args.organization,
            model=args.model_name,
            batch_size=args.batch_size,
        )