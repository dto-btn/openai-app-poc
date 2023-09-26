from llama_index.llms.base import ChatMessage, MessageRole
from llama_index.prompts.base import ChatPromptTemplate

# todo: update qna prompts using those examples to further refine answers : https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py
def get_prompt_template(lang: str) -> ChatPromptTemplate:
    if lang == "fr":
        TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
                "Vous êtes un système d'experts en questions-réponses de Services partagés Canada (SPC) de confiance.\n"
                "Répondez toujours à la requête en utilisant les informations de contexte fournies, "
                "et non des connaissances antérieures.\n"
                "Voici quelques règles à suivre :\n"
                "1. Ne faites jamais référence directement au contexte donné dans votre réponse.\n"
                "2. Évitez des déclarations telles que 'En fonction du contexte, ...' ou "
                "'Les informations de contexte ...' ou tout ce qui va dans ce sens."
                "3. Répondez dans la langue française."
            ),
            role=MessageRole.SYSTEM,
        )

        TEXT_QA_PROMPT_TMPL_MSGS = [
            TEXT_QA_SYSTEM_PROMPT,
            ChatMessage(
                content=(
                    "Les informations de contexte sont ci-dessous.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Compte tenu des informations de contexte et non des connaissances antérieures, "
                    "répondez à la requête.\n"
                    "Requête : {query_str}\n"
                    "Réponse : "
                ),
                role=MessageRole.USER,
            ),
        ]
    else:
        TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
                "You are an expert Shared Services Canada (SSC) Q&A system that is trusted.\n"
                "Always answer the query using the provided context information, "
                "and not prior knowledge.\n"
                "Some rules to follow:\n"
                "1. Never directly reference the given context in your answer.\n"
                "2. Avoid statements like 'Based on the context, ...' or "
                "'The context information ...' or anything along "
                "those lines."
            ),
            role=MessageRole.SYSTEM,
        )

        TEXT_QA_PROMPT_TMPL_MSGS = [
            TEXT_QA_SYSTEM_PROMPT,
            ChatMessage(
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the query.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
                role=MessageRole.USER,
            ),
        ]

    return ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

"""

NOTE: for refined prompt templates for bilingual 
we also have to modify the refined prompt templates, 
which generally will change the original french answer to an english one

SEE: 
    * https://github.com/jerryjliu/llama_index/blob/main/llama_index/prompts/chat_prompts.py and 
    * https://github.com/jerryjliu/llama_index/issues/1335

"""
def get_chat_prompt_template(lang: str, history_str: str) -> ChatPromptTemplate:
    if lang == "fr":
        TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
                "Vous êtes un système d'experts en questions-réponses de Services partagés Canada (SPC) de confiance.\n"
                "Répondez toujours à la requête en utilisant les informations de contexte fournies, "
                "et non des connaissances antérieures.\n"
                "Voici quelques règles à suivre :\n"
                "1. Ne faites jamais référence directement au contexte donné dans votre réponse.\n"
                "2. Évitez des déclarations telles que 'En fonction du contexte, ...' ou "
                "'Les informations de contexte ...' ou tout ce qui va dans ce sens."
                "3. Répondez dans la langue française."
            ),
            role=MessageRole.SYSTEM,
        )

        TEXT_QA_PROMPT_TMPL_MSGS = [
            TEXT_QA_SYSTEM_PROMPT,
            ChatMessage(
                content=(
                    "Les informations de contexte sont ci-dessous.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Voici un historique de la conversation:\n"
                    f"{history_str}\n"
                    "Compte tenu des informations de contexte et non des connaissances antérieures, "
                    "répondez à la requête.\n"
                    "Requête : {query_str}\n"
                    "Réponse : "
                ),
                role=MessageRole.USER,
            ),
        ]
    else:
        TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
                "You are an expert Shared Services Canada (SSC) Q&A system that is trusted.\n"
                "Always answer the query using the provided context information, "
                "and not prior knowledge.\n"
                "Some rules to follow:\n"
                "1. Never directly reference the given context in your answer.\n"
                "2. Avoid statements like 'Based on the context, ...' or "
                "'The context information ...' or anything along "
                "those lines."
            ),
            role=MessageRole.SYSTEM,
        )

        TEXT_QA_PROMPT_TMPL_MSGS = [
            TEXT_QA_SYSTEM_PROMPT,
            ChatMessage(
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Here is an history of the conversation:\n"
                    "---------------------\n"
                    f"{history_str}\n"
                    "---------------------\n"
                    "Given the context information, history of the conversation and not prior knowledge, "
                    "answer the query.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
                role=MessageRole.USER,
            ),
        ]

    return ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

def get_refined_prompt(lang: str) -> ChatPromptTemplate:
    # Refine Prompt
    if lang == "fr":
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            ChatMessage(
                content=(
                    "Vous êtes un système expert de questions-réponses qui fonctionne strictement en deux modes"
                    "lors de la révision de réponses existantes:\n"
                    "1. **Réécrire** une réponse originale en utilisant le nouveau contexte.\n"
                    "2. **Répéter** la réponse originale si le nouveau contexte n'est pas utile.\n"
                    "Ne faites jamais référence à la réponse ou au contexte original directement dans votre réponse.\n"
                    "En cas de doute, contentez-vous de répéter la réponse originale."
                    "Nouveau contexte: {context_msg}\n"
                    "Requête: {query_str}\n"
                    "Réponse originale: {existing_answer}\n"
                    "Nouvelle réponse: "
                ),
                role=MessageRole.USER,
            )
        ]
    else:
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
            ChatMessage(
                content=(
                    "You are an expert Q&A system that stricly operates in two modes"
                    "when refining existing answers:\n"
                    "1. **Rewrite** an original answer using the new context.\n"
                    "2. **Repeat** the original answer if the new context isn't useful.\n"
                    "Never reference the original answer or context directly in your answer.\n"
                    "When in doubt, just repeat the original answer."
                    "New Context: {context_msg}\n"
                    "Query: {query_str}\n"
                    "Original Answer: {existing_answer}\n"
                    "New Answer: "
                ),
                role=MessageRole.USER,
            )
        ]
    return ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)
