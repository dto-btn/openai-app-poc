from langchain import PromptTemplate
from llama_index import QuestionAnswerPrompt, RefinePrompt
from langchain.prompts.chat import (AIMessagePromptTemplate,
                                    ChatPromptTemplate,
                                    HumanMessagePromptTemplate)
from llama_index import QuestionAnswerPrompt

"""

NOTE: for refined prompt templates for bilingual 
we also have to modify the refined prompt templates, 
which generally will change the original french answer to an english one

SEE: 
    * https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py and 
    * https://github.com/jerryjliu/llama_index/issues/1335

"""
def get_chat_prompt_template(lang: str, history_str: str) -> PromptTemplate:
    if lang == "fr":
        QAH_PROMPT_TMPL = (
            "Vous êtes un IA de Services partagés Canada (SPC) propulsée par Azure OpenAI. Nous avons fourni des informations contextuelles ci-dessous.\n"
            "\n---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Voici un historique de la conversation:\n"
            f"{history_str}\n"
            "Humain: {query_str}\n"
            "IA:"
        )
    else:
        QAH_PROMPT_TMPL = (
            "You are a Shared Services Canada (SSC) AI powered by Azure OpenAI. We have provided context information below.\n"
            "\n---------------------\n"
            "{context_str}"
            "\n---------------------\n"
            "Here is an history of the conversation:\n"
            f"{history_str}\n"
            "Human: {query_str}\n"
            "AI:"
        )

    return QuestionAnswerPrompt(QAH_PROMPT_TMPL)

def get_refined_prompt(lang: str):
    # Refine Prompt
    if lang == "fr":
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            HumanMessagePromptTemplate.from_template("{query_str}"),
            AIMessagePromptTemplate.from_template("{existing_answer}"),
            HumanMessagePromptTemplate.from_template(
                "J'ai plus de contexte ci-dessous qui peut être utilisé"
                "(uniquement si nécessaire) pour mettre à jour votre réponse précédente.\n"
                "------------\n"
                "{context_msg}\n"
                "------------\n"
                "Compte tenu du nouveau contexte, mettre à jour la réponse précédente pour mieux"
                "répondre à ma question précédente."
                "Si la réponse précédente reste la même, répétez-la textuellement."
                "Ne référencez jamais directement le nouveau contexte ou ma requête précédente.",
            ),
        ]
    else:
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            HumanMessagePromptTemplate.from_template("{query_str}"),
            AIMessagePromptTemplate.from_template("{existing_answer}"),
            HumanMessagePromptTemplate.from_template(
                "I have more context below which can be used "
                "(only if needed) to update your previous answer.\n"
                "------------\n"
                "{context_msg}\n"
                "------------\n"
                "Given the new context, update the previous answer to better "
                "answer my previous query."
                "If the previous answer remains the same, repeat it verbatim. "
                "Never reference the new context or my previous query directly.",
            ),
        ]

    CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
    return RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)
