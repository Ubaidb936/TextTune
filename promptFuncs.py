def generate_prompt(dataset_point):
    """
    Generate a prompt for a given Hugging Face dataset point.

    Args:
        dataset_point (dict): A dictionary representing a datapoint in a Hugging Face dataset.

    Returns:
        str: A formatted prompt string.
    """
    # Extract context, question, and answer from dataset point
    context = dataset_point["context"]
    question = dataset_point["question"]
    answer = dataset_point["answer"]

    # Define the prompt template
    prompt_template = f"""
    As a expert stock investor, your task is to generate responses to stock investing questions based on the provided context.

    Given the context, generate a response that addresses the question.

    Provide your response as follows:

    Output:::
    Response: (your generated response)

    Here is the stock investing data point:

    Context: {context}
    Question: {question}

    Output:::
    Response: {answer}
    """

    return prompt_template