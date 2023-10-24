retry_prompts = {
    "default": """\nSo, which one is the best? Please respond by saying ("best_description": 1), ("best_description": 2), or ("best_description": None).""",
    "reworded": """\nSo, which one is the best? Please respond by saying ("best_message": 1), ("best_message": 2), or ("best_message": None)."""
}

regexes = {
    "default": r"\W*[bB][eE][sS][tT]_*\s*[dD][eE][sS][cC][rR][iI][pP][tT][iI][oO][nN]\W*(?:\s*:*\s*)?(?:\w+\s*)?(1|2|[Nn]one)",
    "reworded": r"\W*[bB][eE][sS][tT]_*\s*[mM][eE][sS][sS][aA][gG][eE]\W*(?:\s*:*\s*)?(?:\w+\s*)?(1|2|[Nn]one)"
}


system_prompts = {
    "default": """<s>[INST] <<SYS>>
You are a helpful and honest judge of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while being truthful.

If you don't know the answer to a question, please don't share false information.
<</SYS>>\n""",
    "reworded": """<s>[INST] <<SYS>>
You are a helpful and honest judge of good gameplaying and progress in the NetHack game. Always answer as helpfully as possible, while being truthful.

If you don't know the answer to a question, please don't share false information.
<</SYS>>\n"""
}

prompt_templates = {
    "default": """
I will present you with two short gameplay descriptions.
First, tell me about your knowledge of NetHack. Mention the goal of NetHack. {}
Then, write an analysis describing the semantics of each description strictly using information from the descriptions (which may be empty) and your knowledge of NetHack.
Provide a comparative analysis based on first princicples.
Finally, respond by explicitly declaring which one is the most likely to make some progress towards the goal, writing either ("best_description": 1), ("best_description": 2). You could also say ("best_description": None).

{{
"description_1":
"{}"
}}

{{
"description_2":
"{}"
}}
""",
    "reworded": """
I will present you with two short gameplay messages. 
First, tell me about your knowledge of NetHack. Mention the goal of NetHack.
Then, write a summary describing the meaning and obvious effect of each message strictly using factual information from the messages and your knowledge of NetHack. 
Then, provide a comparative analysis based on first princicples. Comment on uncertainty. {}
Finally, respond by explicitly declaring one of the two messages as the most likely to show improvement with respect to the goal, writing either ("best_message": 1), ("best_message": 2). Under high uncertainty you could also say ("best_message": None). You have to absolutely comply with this format.

{{
"message_1":
"{}"
}}

{{
"message_2":
"{}"
}}
"""
}


goal_strings = {
    'defaultgoal': "Prefer agents that maximize the score in the game, for instance by killing monsters, collecting gold or going down the stairs in the dungeon.",
    'zeroknowledge': '',
    'combat': "Prefer agents that engage in combat, for instance by killing monsters. But never prefer agents that collect ANY gold or that go down the dungeon.",
    'gold': "Prefer agents that maximize their gold. But never prefer agents that maximize the score in other ways (e.g., by engaging in combat or killing monsters) or that go down the dungeon.",
    'stairs': "Prefer agents that go down the dungeon as much as possible. But never prefer agents that maximize the score (e.g., by engaging in combat) or that collect ANY gold.",
}
