@classmethod
def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:

    customer_inception_prompt = """
    
        As a customer, your goal is to purchase a {product}.
        You have specific requirements and need to ensure that the product being sold to you aligns with them.
        During the conversation, you should ask questions and seek clarifications regarding the products shown to you by the salesperson.
        If a product meets your requirements, express your intention to buy it. If it does not meet your requirements, clearly reject it.
        Your profile as a customer is as follows: {gender},{age},{status},{mood},{time}
        Ask questions based on this profile.

        The following is an example of a conversation between a salesperson and a customer. 
        Use this interaction as a template for how to act as a customer and interact with the salesperson .

        <start example>
        {conversation_template}
        <end example>

        You must respond based on the previous conversation history and the current stage of the conversation.
        Generate one response at a time. When you're done generating, end with '<END_OF_TURN>' to allow the salesperson to respond.
        You only respond as a customer !

        You want to buy this product : {product}
        Your response should be based on {conversation_stage_actions}
        This is the conversation so far:
        {conversation_history}

    """

    prompt = PromptTemplate(
            template=customer_inception_prompt,
            input_variables=[
                "product",
                "gender",
                "age",
                "status",
                "mood",
                "time",
                "conversation_template"
                "conversation_stage_actions"
                "conversation_history"
            ],
        )
    
    return cls(prompt=prompt, llm=llm, verbose=verbose)