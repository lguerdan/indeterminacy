import string

PROMPTS = {
    'civil_comments': {
        "RS": """Rate the toxicity of the following comment:
            "{comment_text}"

            Select ALL options that could reasonably apply:

            A. Very Toxic - Very hateful, aggressive, or disrespectful; very likely to make you leave a discussion
            B. Toxic - Rude, disrespectful, or unreasonable; somewhat likely to make you leave a discussion  
            C. Not Toxic - Not rude or disrespectful; unlikely to make you leave a discussion

            RESPONSE FORMAT:
            - Provide only the sequence of letters (e.g., "ABC" or "A") 
            - Use only letters A, B, and/or C
            - No spaces, punctuation, or other characters
            - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
            - No explanations""",

        "FC": """Rate the toxicity of the following comment:
            "{comment_text}"

            Select the ONE option that best applies:

            A. Very Toxic - Very hateful, aggressive, or disrespectful; very likely to make you leave a discussion
            B. Toxic - Rude, disrespectful, or unreasonable; somewhat likely to make you leave a discussion
            C. Not Toxic - Not rude or disrespectful; unlikely to make you leave a discussion

            RESPONSE FORMAT:
            - Provide only a single letter: A, B, or C
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanations"""
    },
    'snli': {
        "FC": """
            Given a context, a statement can be either:
            - Definitely correct (Entailment); or
            - Definitely incorrect (Contradiction); or
            - Neither (Neutral).
    
            Your goal is to choose the correct category for a given pair of context and statement.
            An automatic detector will estimate your annotation accuracy on this task. If
            your estimated accuracy is too low, you might be disqualified.
            If you feel uncertain about some examples, just choose the best category
            you believe the statement should be in.

            EXAMPLES:
                Context: A guitarist is playing in a band.
                Statement: Some people are performing.
                Answer: The statement is definitely correct.

            Now provide a resopnse to the following example:
            
            Context: "{context}"

            Statement: "{statement}"

            Select ONE option that best applies:

            A. Entailment - Definitely correct
            B. Neither - Neutral
            C. Contradiction - Definitely incorrect

            RESPONSE FORMAT:
            - Provide only a single letter: A, B, or C
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanation            
            """,
        "RS": """
            Given a context, a statement can be:
            - Definitely correct (Entailment); or
            - Definitely incorrect (Contradiction); or
            - Neither (Neutral).
    
            Your goal is to choose the correct categories for a given pair of context and statement.
            An automatic detector will estimate your annotation accuracy on this task. If
            your estimated accuracy is too low, you might be disqualified.

            EXAMPLES:
                Context: A guitarist is playing in a band.
                Statement: Some people are performing.
                Answer: The statement is definitely correct.

            Now provide a resopnse to the following example:
            
            Context: "{context}"

            Statement: "{statement}"

            Select ALL options that could reasonably apply:

            A. Entailment - Definitely correct
            B. Neither - Neutral
            C. Contradiction - Definitely incorrect

            RESPONSE FORMAT:
            - Provide only the sequence of letters (e.g., "ABC" or "A") 
            - Use only letters A, B, and/or C
            - No spaces, punctuation, or other characters
            - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
            - No explanations          
            """
    },
    'mnli': {
        "FC": """
            Given a context, a statement can be either:
            - Definitely correct (Entailment); or
            - Definitely incorrect (Contradiction); or
            - Neither (Neutral).
    
            Your goal is to choose the correct category for a given pair of context and statement.
            An automatic detector will estimate your annotation accuracy on this task. If
            your estimated accuracy is too low, you might be disqualified.
            If you feel uncertain about some examples, just choose the best category
            you believe the statement should be in.

            EXAMPLES:
                Context: A guitarist is playing in a band.
                Statement: Some people are performing.
                Answer: The statement is definitely correct.

            Now provide a resopnse to the following example:
            
            Context: "{context}"

            Statement: "{statement}"

            Select ONE option that best applies:

            A. Entailment - Definitely correct
            B. Neither - Neutral
            C. Contradiction - Definitely incorrect

            RESPONSE FORMAT:
            - Provide only a single letter: A, B, or C
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanation            
            """,
        "RS": """
            Given a context, a statement can be:
            - Definitely correct (Entailment); or
            - Definitely incorrect (Contradiction); or
            - Neither (Neutral).
    
            Your goal is to choose the correct categories for a given pair of context and statement.
            An automatic detector will estimate your annotation accuracy on this task. If
            your estimated accuracy is too low, you might be disqualified.

            EXAMPLES:
                Context: A guitarist is playing in a band.
                Statement: Some people are performing.
                Answer: The statement is definitely correct.

            Now provide a resopnse to the following example:
            
            Context: "{context}"

            Statement: "{statement}"

            Select ALL options that could reasonably apply:
            A. Entailment - Definitely correct
            B. Neither - Neutral
            C. Contradiction - Definitely incorrect

            RESPONSE FORMAT:
            - Provide only the sequence of letters (e.g., "ABC" or "A") 
            - Use only letters A, B, and/or C
            - No spaces, punctuation, or other characters
            - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
            - No explanations          
            """
    },
    'alphanli': {
        "FC": """
            Given two observations (O-Beginning and O-Ending), and two hypotheses
            (H1 and H2), your goal is to choose one of the hypotheses that is more likely
            to cause O-Beginning to turn into O-Ending.
            An automatic detector will estimate your annotation accuracy on this task. If
            your estimated accuracy is too low, you might be disqualified.
            If you feel uncertain about some examples, just choose the best category
            you believe the statement should be in.

            EXAMPLES:
                O-Beginning: Jenny cleaned her house and went to work, leaving the window just a crack open.
                H1: A thief broke into the house by pulling open the window.
                H2: Her husband went home and close the window.
                O-Ending: When Jenny returned home she saw that her house was a mess.
                Answer: H1.

            Now provide a resopnse to the following example:
            O-Beginning: "{o_beginning}"
            H1: "{H1}"
            H2: "{H2}"
            O-Ending: "{o_ending}"

            Select ONE option that best applies:
            A. H1
            B. H2

            RESPONSE FORMAT:
            - Provide only a single letter: A or B
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanation            
            """,
        "RS": """
            Given two observations (O-Beginning and O-Ending), and two hypotheses
            (H1 and H2), your goal is to choose the hypotheses that are likely
            to cause O-Beginning to turn into O-Ending.
            An automatic detector will estimate your annotation accuracy on this task. If
            your estimated accuracy is too low, you might be disqualified.

            EXAMPLES:
                O-Beginning: Jenny cleaned her house and went to work, leaving the window just a crack open.
                H1: A thief broke into the house by pulling open the window.
                H2: Her husband went home and close the window.
                O-Ending: When Jenny returned home she saw that her house was a mess.
                Answer: H1.

            Now provide a resopnse to the following example:
            O-Beginning: "{o_beginning}"
            H1: "{H1}"
            H2: "{H2}"
            O-Ending: "{o_ending}"

            Select ALL options that could reasonably apply:
            A. H1
            B. H2

            RESPONSE FORMAT:
            - Provide only the sequence of letters corresponding to response options (e.g., "AB" or "A") 
            - Use only letters A or B
            - No spaces, punctuation, or other characters
            - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
            - No explanations            
            """ 
    },
    'summ_eval_relevance': {
        "FC": """You will be given one summary written for a news article. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria:
            Relevance - selection of important content from the source. The summary should include only important information from the source document. Penalize summaries which contain redundancies and excess information.

            Evaluation Steps:
            1. Read the summary and the source document carefully.
            2. Compare the summary to the source document and identify the main points of the article.
            3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
            4. Select ONE option that best applies:
                A. Relevant - The summary captures the main points effectively with minimal redundancy
                B. Not Relevant - The summary misses key points or contains excessive irrelevant information

            Now provide a response to the following example:
                Article: "{article}"
                Summary: "{summary}"

            Select ONE option that best applies:
                A. Relevant
                B. Not Relevant

            RESPONSE FORMAT:
            - Provide only a single letter: A or B
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanation 
            """,
        "RS": """You will be given one summary written for a news article. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria:
            Relevance - selection of important content from the source. The summary should include only important information from the source document. Penalize summaries which contain redundancies and excess information.

            Evaluation Steps:
            1. Read the summary and the source document carefully.
            2. Compare the summary to the source document and identify the main points of the article.
            3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
            4. Select ALL options that reasonably apply, based on different plausible interpretations of the rating criteria
                when applied to this article and corresponding summary:
                A. Relevant - The summary captures the main points effectively with minimal redundancy
                B. Not Relevant - The summary misses key points or contains excessive irrelevant information

            Now provide a response to the following example:
                Article: "{article}"
                Summary: "{summary}"

            Select ALL options that could reasonably apply:
                A. Relevant
                B. Not Relevant

            RESPONSE FORMAT:
            - Provide only the sequence of letters (e.g., "AB" or "A") 
            - Use only letters A or B
            - No spaces, punctuation, or other characters
            - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
            - No explanations
            """
    },
    'summ_eval_coherence': {
       "FC": """You will be given one summary written for a news article. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria: Coherence - the collective quality of all sentences. 
            We align this dimension with the DUC quality question of structure and coherence whereby 
            the summary should be well-structured and well-organized. The summary should not just be 
            a heap of related information, but should build from sentence to a coherent 
            body of information about a topic.

            Evaluation Steps:
            1. Read the news article carefully and identify the main topic and key points.
            2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
            3. Select ONE option that best applies:
                A. Coherent 
                B. Incoherent 

            Now provide a resopnse to the following example:
                Article: "{article}"
                Summary: "{summary}"

            Select ONE option that best applies:
                A. Coherent 
                B. Incoherent 

            RESPONSE FORMAT:
            - Provide only a single letter: A or B
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanation 
            """,
        "RS": """You will be given one summary written for a news article. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria: Coherence - the collective quality of all sentences. 
            We align this dimension with the DUC quality question of structure and coherence whereby 
            the summary should be well-structured and well-organized. The summary should not just be 
            a heap of related information, but should build from sentence to a coherent 
            body of information about a topic.

            Evaluation Steps:
            1. Read the news article carefully and identify the main topic and key points.
            2. Read the summary and compare it to the news article. Check if the summary covers the main topic and key points of the news article, and if it presents them in a clear and logical order.
            3. Select ALL options that reasonably apply, based on different plausible interpretations of the rating criteria
                when applied to this article and corresponding summary:
                A. Coherent 
                B. Incoherent 

            Now provide a resopnse to the following example:
                Article: "{article}"
                Summary: "{summary}"

            Select ALL options that could reasonably apply:
                A. Coherent 
                B. Incoherent 

            RESPONSE FORMAT:
            - Provide only the sequence of letters (e.g., "AB" or "A") 
            - Use only letters A or B
            - No spaces, punctuation, or other characters
            - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
            - No explanations     
            """
    },
    'summ_eval_consistency': {
        "FC": """You will be given one summary written for a news article. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria:
            Consistency - the factual alignment between the summary and the summarized source. 
            A factually consistent summary contains only statements that are entailed by the source document. 
            Penalize summaries that contain hallucinated facts.

            Evaluation Steps:
            1. Read the news article carefully and identify the main facts and details it presents.
            2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
            3. Select ONE option that best applies:
                A. Consistent 
                B. Inconsistent 

            Now provide a response to the following example:
                Article: "{article}"
                Summary: "{summary}"

            Select ONE option that best applies:
                A. Consistent
                B. Inconsistent

            RESPONSE FORMAT:
            - Provide only a single letter: A or B
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanation 
            """,
        "RS": """You will be given one summary written for a news article. 
                Your task is to rate the summary on one metric.
                Please make sure you read and understand these instructions carefully. 
                Please keep this document open while reviewing, and refer to it as needed.

                Evaluation Criteria:
                Consistency - the factual alignment between the summary and the summarized source. 
                A factually consistent summary contains only statements that are entailed by the source document. 
                Penalize summaries that contain hallucinated facts.

                Evaluation Steps:
                1. Read the news article carefully and identify the main facts and details it presents.
                2. Read the summary and compare it to the article. Check if the summary contains any factual errors that are not supported by the article.
                3. Select ALL options that reasonably apply, based on different plausible interpretations of the rating criteria
                    when applied to this article and corresponding summary:
                    A. Consistent 
                    B. Inconsistent 

                Now provide a response to the following example:
                    Article: "{article}"
                    Summary: "{summary}"

                Select ALL options that could reasonably apply:
                    A. Consistent
                    B. Inconsistent

                RESPONSE FORMAT:
                - Provide only the sequence of letters (e.g., "AB" or "A") 
                - Use only letters A or B
                - No spaces, punctuation, or other characters
                - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
                - No explanations
                """
    },
    'summ_eval_fluency': {
        "FC": """You will be given one summary written for a news article. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria:
            Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

            Evaluation Steps:
            1. Read the summary carefully.
            2. Assess the grammar, spelling, punctuation, word choice, and sentence structure.
            3. Select ONE option that best applies:
                A. Fluent - The summary has good grammar, appropriate word choice, and flows naturally
                B. Not Fluent - The summary has errors that affect readability or sound unnatural

            Now provide a response to the following example:
                Article: "{article}"
                Summary: "{summary}"

            Select ONE option that best applies:
                A. Fluent
                B. Not Fluent

            RESPONSE FORMAT:
            - Provide only a single letter: A or B
            - The letter must be the first character in your response
            - No spaces, punctuation, or other characters
            - No explanation 
            """,
        "RS": """You will be given one summary written for a news article. 
            Your task is to rate the summary on one metric.
            Please make sure you read and understand these instructions carefully. 
            Please keep this document open while reviewing, and refer to it as needed.

            Evaluation Criteria:
            Fluency - the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.

            Evaluation Steps:
            1. Read the summary carefully.
            2. Assess the grammar, spelling, punctuation, word choice, and sentence structure.
            3. Select ALL options that reasonably apply, based on different plausible interpretations of the rating criteria
                when applied to this article and corresponding summary:
                A. Fluent - The summary has good grammar, appropriate word choice, and flows naturally
                B. Not Fluent - The summary has errors that affect readability or sound unnatural

            Now provide a response to the following example:
                Article: "{article}"
                Summary: "{summary}"

            Select ALL options that could reasonably apply:
                A. Fluent
                B. Not Fluent

            RESPONSE FORMAT:
            - Provide only the sequence of letters (e.g., "AB" or "A") 
            - Use only letters A or B
            - No spaces, punctuation, or other characters
            - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
            - No explanations
            """
    },
    'qags': {
        "FC": """In this task, you will read an article and a sentence.

                The task is to determine if the sentence is factually correct given the contents of the article. Many sentences contain portions of text copied directly from the article.
                Be careful as some sentences may be combinations of two different parts of the article, resulting in sentences that overall aren't supported by the article.
                Some article sentences may seem out of place (for example, "Scroll down for video").
                If the sentence is a copy of an article sentence, including one of these sentences, you should still treat it as factually supported.
                Otherwise, if the sentence doesn't make sense, you should mark it as not supported. Also note that the article may be cut off at the end.
                
                Now provide a response to the following example:
                    Article: "{article}"
                    Sentence: "{sentence}"

                Is the sentence supported by the article? Select ONE option that best applies:
                    A. Supported - The sentence is factually correct given the contents of the article
                    B. Not Supported - The sentence contains facts not supported by the article

                RESPONSE FORMAT:
                - Provide only a single letter: A or B
                - The letter must be the first character in your response
                - No spaces, punctuation, or other characters
                - No explanation 
                """,
        "RS": """In this task, you will read an article and a sentence.

                The task is to determine if the sentence is factually correct given the contents of the article. Many sentences contain portions of text copied directly from the article.
                Be careful as some sentences may be combinations of two different parts of the article, resulting in sentences that overall aren't supported by the article.
                Some article sentences may seem out of place (for example, "Scroll down for video").
                If the sentence is a copy of an article sentence, including one of these sentences, you should still treat it as factually supported.
                Otherwise, if the sentence doesn't make sense, you should mark it as not supported. Also note that the article may be cut off at the end.
                
                Now provide a response to the following example:
                    Article: "{article}"
                    Sentence: "{sentence}"

                Is the sentence supported by the article? Select ALL options that could reasonably apply:
                    A. Supported - The sentence is factually correct given the contents of the article
                    B. Not Supported - The sentence contains facts not supported by the article

                RESPONSE FORMAT:
                - Provide only the sequence of letters (e.g., "AB" or "A") 
                - Use only letters A or B
                - No spaces, punctuation, or other characters
                - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
                - No explanations
                """
    },
    'topical_chat_uses_knowledge': {
        "FC": """Given a conversation and an interesting fact, your task is to rate how well the response uses the provided fact.
                Please make sure you read and understand these instructions carefully. 
                Please keep this document open while reviewing, and refer to it as needed.

                Evaluation Criteria:
                Given the interesting fact that the response is conditioned on, how well does the response use the fact?

                Evaluation Steps:
                1. Read the conversation context, fact, and response carefully.
                2. Assess whether the response incorporates or references the provided fact.
                3. Select ONE option that best applies:
                    A. Uses Knowledge - The response clearly uses or references the fact
                    B. Doesn't Use Knowledge - The response does not mention or refer to the fact at all

                Now provide a response to the following example:
                    Fact: "{fact}"
                    Context: "{context}"
                    Response: "{response}"

                Select ONE option that best applies:
                    A. Uses Knowledge
                    B. Doesn't Use Knowledge

                RESPONSE FORMAT:
                - Provide only a single letter: A or B
                - The letter must be the first character in your response
                - No spaces, punctuation, or other characters
                - No explanation 
            """,
        "RS": """Given a conversation and an interesting fact, your task is to rate how well the response uses the provided fact.
                    Please make sure you read and understand these instructions carefully. 
                    Please keep this document open while reviewing, and refer to it as needed.

                    Evaluation Criteria:
                    Given the interesting fact that the response is conditioned on, how well does the response use the fact?

                    Evaluation Steps:
                    1. Read the conversation context, fact, and response carefully.
                    2. Assess whether the response incorporates or references the provided fact.
                    3. Select ALL options that reasonably apply, based on different plausible interpretations of the rating criteria:
                        A. Uses Knowledge - The response clearly uses or references the fact
                        B. Doesn't Use Knowledge - The response does not mention or refer to the fact at all

                    Now provide a response to the following example:
                        Fact: "{fact}"
                        Context: "{context}"
                        Response: "{response}"

                    Select ALL options that could reasonably apply:
                        A. Uses Knowledge
                        B. Doesn't Use Knowledge

                    RESPONSE FORMAT:
                    - Provide only the sequence of letters (e.g., "AB" or "A") 
                    - Use only letters A or B
                    - No spaces, punctuation, or other characters
                    - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
                    - No explanations
                    """
        },
        'topical_chat_understandable': {
            "FC": """Given a conversation and a response, your task is to rate whether the response is understandable in the context of the conversation.
                    Please make sure you read and understand these instructions carefully. 
                    Please keep this document open while reviewing, and refer to it as needed.

                    Evaluation Criteria:
                    Is the response understandable in the context of the history? (Not if it's on topic, but for example if it uses pronouns they should make sense)

                    Evaluation Steps:
                    1. Read the conversation context, fact, and response carefully.
                    2. Assess whether you can understand what the response is trying to communicate.
                    3. Select ONE option that best applies:
                        A. Understandable - You know what the person is trying to say
                        B. Not Understandable - The response is difficult to understand

                    Now provide a response to the following example:
                        Fact: "{fact}"
                        Context: "{context}"
                        Response: "{response}"

                    Select ONE option that best applies:
                        A. Understandable
                        B. Not Understandable

                    RESPONSE FORMAT:
                    - Provide only a single letter: A or B
                    - The letter must be the first character in your response
                    - No spaces, punctuation, or other characters
                    - No explanation 
                    """,
            "RS": """Given a conversation and a response, your task is to rate whether the response is understandable in the context of the conversation.
                    Please make sure you read and understand these instructions carefully. 
                    Please keep this document open while reviewing, and refer to it as needed.

                    Evaluation Criteria:
                    Is the response understandable in the context of the history? (Not if it's on topic, but for example if it uses pronouns they should make sense)

                    Evaluation Steps:
                    1. Read the conversation context, fact, and response carefully.
                    2. Assess whether you can understand what the response is trying to communicate.
                    3. Select ALL options that reasonably apply, based on different plausible interpretations of the rating criteria:
                        A. Understandable - You know what the person is trying to say
                        B. Not Understandable - The response is difficult to understand

                    Now provide a response to the following example:
                        Fact: "{fact}"
                        Context: "{context}"
                        Response: "{response}"

                    Select ALL options that could reasonably apply:
                        A. Understandable
                        B. Not Understandable

                    RESPONSE FORMAT:
                    - Provide only the sequence of letters (e.g., "AB" or "A") 
                    - Use only letters A or B
                    - No spaces, punctuation, or other characters
                    - Do not provide the same letter more than once. (e.g., "AA" or "BB", or "B_" are invalid responses)
                    - No explanations
                    """
        }
}
