def inferance(query: str, model, tokenizer, temp = 1.0, limit = 200) -> str:
    
    device = "cuda:0"

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

    """
    prompt = prompt_template.format(query=query)

    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)

    model_inputs = encodeds.to(device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=int(limit), temperature=temp, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids)
    return (decoded[0])