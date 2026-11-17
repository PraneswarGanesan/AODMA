
from agents.llm import get_llm, load_llm_config


def main():
    config = load_llm_config()

    print("AODMA LLM CONNECTION TEST")
    print("-------------------------")
    print("Provider:", config.provider)
    print("Model:", config.model)

    llm = get_llm()

    response = llm.invoke(
        "You are the AODMA Research Agent. "
        "Explain your role in one sentence."
    )

    print("\nMistral response:")
    print(response.content)


if __name__ == "__main__":
    main()
