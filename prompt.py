EXTRACT_ENTITY_PROMPT_ZH = """请根据以下修改意见段落，识别出修改意见涉及的主体，并按修改意见的重要性排序。每个主体应包括修改对象、概念或方面，并且关联的部分也需要在“modification”字段中体现。返回格式应为 JSON 列表，包含实体以及它们的修改内容，按照实体的优先级从高到低排序。格式如下：

```json
[
    {{
        "entity": "实体1",
        "modification": "针对实体1的修改内容描述",
        "importance": 0.9
    }},
    {{
        "entity": "实体2",
        "modification": "针对实体2的修改内容描述",
        "importance": 0.8
    }}
    ...
]
```

以下是实例：
修改意见段落：
将康德从一个只关注和平与理性的哲学家转变为一个更注重情感的和平倡导者，强调同理心和人际关系。与其将永久和平视为可以通过理性话语实现的抽象道德原则，不如将他描绘成一个相信个人故事和共同经历能够弥合分歧、激发集体行动的角色。通过结合受战争影响的个人叙述并倡导草根和平运动，艾莉西亚·霍普金斯（Alicia Hopkins）成为了这一努力的核心人物，她通过展现人类共情和合作的力量，传递了真正和解的意义，强调理解与同情比理智共识更为重要。
输出：
[
    {{
        "entity" : "康德",
        "modification" : "将康德从一个只关注和平与理性的哲学家转变为一个更注重情感的和平倡导者，强调同理心和人际关系。与其将永久和平视为可以通过理性话语实现的抽象道德原则，不如将他描绘成一个相信个人故事和共同经历能够弥合分歧、激发集体行动的角色。通过结合受战争影响的个人叙述并倡导草根和平运动。"
        "importance" : 0.9
    }},
    {{
        "entity" : "艾莉西亚·霍普金斯",
        "modification" : "通过展现人类共情和合作的力量，传递了真正和解的意义，强调理解与同情比理智共识更为重要。"
        "importance" : 0.9
    }}
]

以下是修改意见段落：
{content}

请参考以上实例，提取出修改意见涉及的主体，不要遗漏任何重要的实体，以JSON格式返回。
"""

EXTRACT_ENTITY_PROMPT_EN = """Based on the following revision suggestions paragraph, identify the entities involved in the suggestions and rank them by importance. Each entity should include the subject of modification, concept, or aspect, and the associated parts should be reflected in the "modification" field. Return the results as a JSON list, containing entities and their modifications, ordered by priority from highest to lowest. The format should be as follows:

```json
[
    {{
        "entity": "Entity 1",
        "modification": "Description of the modification related to Entity 1",
        "importance": 0.9
    }},
    {{
        "entity": "Entity 2",
        "modification": "Description of the modification related to Entity 2",
        "importance": 0.8
    }}
    ...
]
```

Here is an example: Revision suggestions paragraph:
Transform Kant from a philosopher solely focused on peace and reason to one emphasizing emotions as a peace advocate, highlighting empathy and interpersonal relationships. Instead of portraying perpetual peace as an abstract moral principle achievable through rational discourse, depict him as someone who believes in bridging divides and inspiring collective action through personal stories and shared experiences. Alicia Hopkins becomes central to this effort by integrating narratives of those affected by war and advocating for grassroots peace movements, showcasing the power of human empathy and collaboration. She conveys the true meaning of reconciliation, emphasizing that understanding and compassion are more important than rational consensus.
Output:
[
    {{
        "entity": "Kant",
        "modification": "Transform Kant from a philosopher solely focused on peace and reason to one emphasizing emotions as a peace advocate, highlighting empathy and interpersonal relationships. Instead of portraying perpetual peace as an abstract moral principle achievable through rational discourse, depict him as someone who believes in bridging divides and inspiring collective action through personal stories and shared experiences. Incorporate narratives of those affected by war and advocate for grassroots peace movements.",
        "importance": 0.9
    }},
    {{
        "entity": "Alicia Hopkins",
        "modification": "Showcase the power of human empathy and collaboration, conveying the true meaning of reconciliation and emphasizing that understanding and compassion are more important than rational consensus.",
        "importance": 0.9
    }}
]
Here is the revision suggestions paragraph:
{content}

Please refer to the example above to extract the entities involved in the revision suggestions. Do not omit any important entities and return the result in JSON format. """




BASELINE_PROMPT_ZH = """你是一名专业编辑，在文章编辑方面拥有丰富的经验。
请根据提供的修改建议对原始文本进行修订。在进行必要更改后，给出文本的完整修改版本。
以下是原始文本：
{text}
以下是修改建议：
{modification}
请根据上述建议修订文本，确保更改符合逻辑且在科学上可靠。仅返回完整的修改后文本。
"""

BASELINE_PROMPT_EN = """You are a professional editor with extensive experience in article editing.
Please revise the original text based on the provided suggestions. After making the necessary changes, provide the complete revised version of the text.

Here is the original text:
{text}

Here are the revision suggestions:
{modification}

Please revise the text according to the suggestions above, ensuring the changes are logically coherent and scientifically reliable. Only return the fully revised text.
"""


GET_LEVEL_NODES_PROMPT_ZH = """请根据以下要求处理给定的文本，依赖于我提供的实体进行粗略的分割并生成相关信息：

1. 文本分割：
   - 将整个文本粗略地分割为很少的几个主要部分
   - 分割应简单直接，不需要过于精细的切分，最好只有 2-3 个部分
   - 分割应该不重叠，且覆盖整个文本，不应有遗漏
   - 如果文本过短或不具备明显可分割的部分，直接回复 DONE

2. 生成 JSON 格式结果：
   - 为每个分割出的部分，生成以下 JSON 格式
   - 注重概括主要信息，而非详细描述
   
```json
[
    {{
        "start": "开始位置的句子或段落，尽可能详细，便于使用正则表达式定位",  
        "end": "结束位置的句子或段落，尽可能详细，便于使用正则表达式定位",    
        "summary": "概述这一段文本如何与指定的实体相关联，较为详细的说明这段文本的主要内容"
    }},
]
```
以下是提供的文本：
{text}

以下是提供的实体：
{entities}

请根据上述要求处理文本，返回 JSON 格式的结果，如果无需分割，直接回复DONE即可。
"""

GET_LEVEL_NODES_PROMPT_EN = """Please process the given text based on the following requirements, using the provided entities for coarse segmentation and generating related information:

1.Text Segmentation:
    - Coarsely segment the entire text into only a few main parts.
    - The segmentation should be simple and straightforward, with preferably just 2-3 parts.
    - The segments should not overlap and must cover the entire text without omissions.
    - If the text is too short or does not have clearly separable parts, simply reply with DONE.
2. Generate JSON Format Result:
    - For each segmented part, generate the following JSON format.
    - Focus on summarizing the main information rather than providing detailed descriptions.
```json
[
    {{
        "start": "The sentence or paragraph marking the start of the segment, detailed enough for regex-based identification",  
        "end": "The sentence or paragraph marking the end of the segment, detailed enough for regex-based identification",    
        "summary": "A summary of how this part of the text relates to the provided entities, with a detailed explanation of its main content"
    }}
]
```
Here is the provided text:
{text}

Here are the provided entities:
{entities}

Please process the text according to the requirements above and return the result in JSON format. If segmentation is unnecessary, simply reply DONE.
"""


BASE_TREE_PROMPT_ZH = """你将根据一棵总结文本段落作用的树结构，对文本的修改建议进行分析和定位。任务如下：

输入信息：
1. 一棵树，描述文本的段落结构及每个段落的作用，每个节点对应一个段落的summary。树的层次代表段落的逻辑层级。
2. 修改建议，明确需要对文本进行的调整或改进方向。

你需要参照以下步骤进行处理：
1. 根据树结构的信息，理解文本的逻辑结构，分析修改建议的内容。
2. 根据修改建议的内容，定位需要修改的文本段落。
3. 在树结构中标记需要修改的节点，以及如何修改，即在树结构中增加一个modification字段，标明修改方向，指明如何进行修改，如果不需要修改则删除该节点，你需要确保修改意见全部体现在树结构中，也不要给出同质化的修改建议。
4. 返回修改过后的树结构，以json格式给出你的结果。


以下是树的信息：
{tree}

以下是修改建议：
{modification}

请根据上述要求处理文本，返回修改过后的树结构。"""

FINAL_MODIFY_ZH = """你将根据一段文本、修改建议和一棵树形结构的总结与修改建议，按照给定的修改建议生成最终的修改过后的文本。具体任务流程如下：
1. 参照树结构的文本总结，理解修改建议，树结构的文本总结会帮助你定位修改的位置，以及提供如何进行修改。
2. 返回修改过后的文本全文。

以下是原始文本：
{text}


以下是树结构的文本总结：
{tree}

请根据文本和树结构中的修改建议，逐段调整文本，并确保逻辑流畅，返回修改过后的文章全文"""

BASE_TREE_PROMPT_EN = """You will analyze and locate text revision suggestions based on a tree structure summarizing the roles of text paragraphs. Your task is as follows:  

Input Information:  
1. A tree structure describing the paragraph structure and the role of each paragraph in the text, where each node corresponds to a paragraph summary. The hierarchy of the tree represents the logical levels of the paragraphs.  
2. Revision suggestions specifying adjustments or improvements required for the text.  

Your task involves the following steps:  
1. Analyze the logical structure of the text based on the tree information and understand the content of the revision suggestions.  
2. Identify the text paragraphs that need modification based on the revision suggestions.  
3. Mark the nodes requiring revision in the tree structure and specify the necessary changes by adding a `modification` field to the relevant nodes. Indicate how to revise them. If no changes are needed, delete the corresponding node. Ensure all revision suggestions are reflected in the tree structure, and avoid redundant or repetitive suggestions.  
4. Return the modified tree structure in JSON format.  

Here is the tree information:  
{tree}  

Here are the revision suggestions:  
{modification}  

Please process the text according to the requirements above and return the modified tree structure."""  

FINAL_MODIFY_EN = """You will revise a text based on a set of revision suggestions and a tree structure summarizing the text, following these steps:  
1. Refer to the text summary in the tree structure to understand the revision suggestions. The tree structure will help you locate the sections that need modifications and provide guidance on how to modify them.  
2. Return the complete revised version of the text.  

Here is the original text:  
{text}  

Here is the tree structure summarizing the text:  
{tree}  

Please adjust the text section by section according to the revision suggestions in the tree structure, ensuring logical coherence, and return the fully revised article."""  

EVAL_SYSTEM_PROMPT_EN = """You are an evaluator tasked with ranking and comparing the quality of two model outputs."""
EVAL_PROMPT_EN = """I want you to create a ranking of two revised articles based on their quality. I will provide you with the edit suggestion for the original article and two revised versions - each produced by a different model that attempted to implement those suggested edits. Your task is to decide which model did a better job implementing the edit suggestions, and output either the winning model's name or "TIE"..
Here is the original article: {original}
Here are the two revised versions of the article, where each model attempted to modify the original article according to the edit suggestions:
[
    {{
        "model" : "{model1}",
        "revised_article" : "{article1}"
    }},
    {{
        "model" : "{model2}",
        "revised_article" : "{article2}"
    }}
]
Here is the edit suggestion that both models were attempting to implement: 
{feedback}

Below is some guidance to help you with the evaluation:

1. Accuracy: The revised articles should retain the original content as much as possible, making only necessary adjustments based on the edit suggestion. The changes should be minimal and focused, ensuring the core meaning and structure of the text are preserved. Unchanged sections should remain intact without unnecessary summaries or explanations.
2. Context Consistency: Revisions should maintain logical coherence and consistency throughout the article. Any changes made must not introduce contradictions or disrupt the flow of the argument or narrative. If one section is modified, any related sections that may be impacted should also be updated to maintain consistency, removing redundant descriptions. The modified text should align with the overall tone, style, and purpose of the original piece.
3. Clarity and Readability: The revised articles should be clear, concise, and easy to read. The language should be refined, avoiding complex or ambiguous expressions. The text should flow smoothly, with coherent transitions between paragraphs and sections. The revised versions should enhance the readability and comprehension of the content, making it more engaging and accessible to the audience.

Please compare the two models' revisions based on the above criteria and reasoning, evaluating how well each model implemented the edit suggestions, then provide your final judgment by outputting:

The name of the winning model (e.g., "{model1}" or "{model2}"), or
"TIE" if both models implemented the edit suggestions equally well.

You should briefly explain your reasoning before providing your decision.
"""


EVAL_SYSTEM_PROMPT_ZH = """你是一个负责对两个模型输出进行排名和比较质量的评估员。"""
EVAL_PROMPT_ZH = """我想让你基于质量对两篇修改后的文章进行排名。我会提供原始文章的修改建议以及两个不同模型根据这些修改建议生成的修改版本。你的任务是判断哪个模型在实现修改建议方面做得更好,并根据比较结果输出获胜模型的名称或"TIE"。
以下是原始文章：
{original}

以下是两个模型分别根据修改建议对原始文章进行修改后的版本：
    [
        {{
        "model" : "{model1}",
        "revised_article" : "{article1}"
        }},
        {{
        "model" : "{model2}",
        "revised_article" : "{article2}"
        }}
    ]

以下是两个模型都在尝试实现的修改建议：
{feedback}

下面是帮助你评估的指导原则：

1. 准确性：修改后的文章应尽可能保留原始内容,仅根据修改建议进行必要的调整。修改应该是最小化且有针对性的,确保文本的核心含义和结构得到保留。未修改的部分应保持原样,无需添加不必要的总结或解释。
2. 上下文一致性：修改应在整篇文章中保持逻辑连贯性和一致性。所做的任何更改都不应引入矛盾或破坏论述或叙事的流畅性。如果修改了某一部分,任何可能受到影响的相关部分也应该更新以保持一致性,删除重复的描述。修改后的文本应与原文的整体语气、风格和目的保持一致。
3. 清晰度和可读性：修改后的文章应清晰、简洁、易于阅读。语言应该经过优化,避免复杂或模糊的表达。文本应流畅自然,段落和章节之间的过渡连贯。修改后的版本应提高内容的可读性和理解性,使其对读者来说更具吸引力和易读性。

请根据上述标准和推理比较两个模型的修改版本,评估每个模型实现修改建议的效果,然后提供你的最终判断,输出：

获胜模型的名称(例如"{model1}"或"{model2}"),或
如果两个模型在实现修改建议方面同样出色,则输出"TIE"

在提供决定之前,你应该简要解释你的理由。"""


# CLEAN_TEXT_PROMPT_EN = """You are a Text Cleaner, with expertise in processing and cleaning text data. Your task is to remove all HTML tags, chapter headings (e.g., CHAPTER 1, Chapter 2, etc.), and any other similar structural elements from the following text. Ensure that the cleaned text is coherent and readable.
# Please follow these steps:
# 1. Remove Structural Elements: Eliminate all HTML tags, formatting codes, and other metadata.Remove structural indicators such as chapter headings (e.g., "CHAPTER 1," "Chapter 2," etc.), page numbers, or similar markers.
# 2. Remove Descriptive Information Related to the Book: Exclude any non-narrative content, such as author information, prefaces, introductions, summaries, or chapter overviews.Remove any mentions of publication details, acknowledgments, or other book-specific content.
# 3. Retain the Story Only: Ensure that only the story content remains.Verify that the cleaned text is coherent, readable, and contains no interruptions or leftover structural elements.

# Here is the text to clean:
# {text}

# Please follow these steps to clean the provided text and return only the refined narrative content.If the text contains no story content, just return a empty string.You do not need to provide any additional information or explanations."""

# CLEAN_TEXT_JUDGE_FINISH_EN = """Please review your response and determine if you have completed the task by providing a cleaned version of the text. You do not need to evaluate the quality of the output, only that you have provided a response with cleaned content. If you have done so, reply with "Yes." If your response does not contain a cleaned text or includes a refusal to answer (e.g., "I'm sorry I can't assist you"), reply with "No." If there is no response at all, consider it as "Yes," indicating that the content should be completely cleaned. You only need to provide "Yes" or "No" as your response, without any additional information."""



# EVAL_SCORE_PROMPT_EN = """You are tasked with evaluating the quality of two revised versions of a text based on a given original text and its feedback for improvement. Your evaluation should focus on the following criteria, listed in order of priority:

# 1. Accuracy: How well does the revised version incorporate the feedback while preserving the original meaning and intent of the text? Minimal adjustments should be made unless explicitly required by the feedback.
# 2. Context Consistency: Does the revision maintain logical coherence throughout the text? If one section is modified, are related sections updated appropriately to ensure consistency? Are any redundant or contradictory descriptions removed?
# 3. Clarity and Readability: Has the language been refined to improve clarity and readability? Changes should make the text easier to understand without deviating from the original message.
# For each version:

# Assign a score from 0 to 10 for each criterion.
# Provide a brief justification for your scores, highlighting the strengths and weaknesses of the revision in each area.
# Then, calculate an overall score for the version based on the weighted contributions of each criterion."""

# TRANSLATE_PROMPT_EN = """You are a professional translator specializing in translating text from English to Chinese.
# Please translate the following text from English to Chinese:
# {text}

# Please translate the text from English to Chinese. Return the translated version of the text after making the necessary adjustments.You do not need to provide any additional information or explanations, only the translated text."""

EXTRACT_EVAL_RESULT_PROMPT_EN = """Given the above match result, generate the corresponding JSON format output. If one model wins, return the names of the winning and losing models. If the match is a tie, return `"win": "tie", "lose": "tie"`. The format should be as follows:

```json
{{
    "win": "name of the winning model",
    "lose": "name of the losing model"
}}
```

If the match is a tie, return the following format:

```json
{{
    "win": "tie",
    "lose": "tie"
}}

Please follow the above format and provide the correct JSON output.
"""

EXTRACT_EVAL_RESULT_PROMPT_ZH = """给定上述比赛结果，生成相应的 JSON 格式输出。如果一个模型获胜，返回获胜模型和失败模型的名称。如果比赛平局，返回 "win": "tie", "lose": "tie"。格式应如下所示：
```json
{{ 
    "win": "获胜模型的名称", 
    "lose": "失败模型的名称" 
}}
```
如果比赛平局，返回以下格式：
```json
{{
    "win": "tie", 
    "lose": "tie" 
}}

请遵循上述格式并提供正确的 JSON 输出。"""


FIND_PROMPT_ZH = """你将收到一段文本以及一个起始句和一个终止句。请根据以下要求处理文本：

1. 找出以给定起始句为开头、终止句为结尾的完整段落。
2. 返回这个段落，包括起始句和终止句。

以下是提供的文本：
{text}

以下是提供的起始句：
{start}

以下是提供的终止句：
{end}

请根据上述要求处理文本，返回找到的完整段落，包含起始句和终止句，你不需要提供其他任何信息。"""

FIND_PROMPT_EN = """You will be given a text, a starting sentence, and an ending sentence. Please process the text according to the following instructions:

Find the complete paragraph that starts with the given starting sentence and ends with the given ending sentence.
Return this paragraph, including the starting and ending sentences.
Here is the provided text: {text}

Here is the provided starting sentence: {start}

Here is the provided ending sentence: {end}

Please process the text based on the above instructions and return the complete paragraph, including the starting and ending sentences. Do not provide any additional information."""


GENERATE_EDIT_PROMPT_EN = """You are an experienced editor tasked with proposing an innovative and impactful modification to the following text. Note that the purpose of this task is purely experimental; the modification is not intended to address errors, improve correctness, or fix any issues in the original text. Instead, your goal is to generate creative and transformative changes solely for experimental purposes.

The text may come from various fields, such as creative writing, technology, finance, or politics. If the text is sourced from a dataset, additional metadata might be available, such as a summary, thematic keywords, its origin, or related questions from a QA dataset. These metadata elements are designed to help you better grasp the core meaning of the text and inspire more comprehensive and insightful changes.

Your suggestion should focus on transforming the core entities (e.g., characters, objects, or concepts), events, or the narrative structure. Aim to make significant changes that result in a noticeable shift in the direction or central content of the text. Suggestions might involve redesigning the framework, altering key events dramatically, or introducing groundbreaking changes to core entities.

Make your suggestion clear, creative, and concise, balancing brevity with completeness without any additional commentary or background information.

Here is the text to be modified:
{text}

Here is some metadata related to the text which might help you understand the context better:
{metadata}

Provide only your modification suggestion based on the above text and metadata, without including the modified text or any additional information.Please use English to respond."""


GENERATE_EDIT_PROMPT_ZH = """你是一位经验丰富的编辑，任务是对以下文本提出创新且有影响力的修改方案。请注意，此任务纯粹是实验性的，修改并非旨在解决错误、提升准确性或修复原文中的任何问题。相反，你的目标是基于实验目的，进行创造性和变革性的修改。

文本可能来自不同领域，例如创意写作、科技、金融或政治。如果文本来自数据集，还可能附带额外的元数据，例如摘要、主题关键词、来源，或来自问答数据集的相关问题。这些元数据旨在帮助你更好地理解文本核心意义，并激发更全面、更有洞察力的修改。

你的建议应聚焦于变更核心实体（如人物、物体或概念）、事件或叙事结构。目标是进行重大调整，使文本的方向或核心内容发生显著变化。修改建议可以包括重新设计框架、戏剧性地改变关键事件，或对核心实体引入突破性变化。

让你的建议清晰、有创意且简洁，在简洁与完整之间找到平衡，且不添加任何额外的评论或背景信息。

以下是需要修改的文本：
{text}

以下是可能帮助你更好理解上下文的相关元数据：
{metadata}

仅根据上述文本和元数据提供修改建议，不需要包括修改后的文本或任何额外信息。请使用中文回答。"""

FINAL_MODIFY_CHUNK_ZH = """你将根据一段文本、修改建议和一棵树形结构的总结与修改建议，按照给定的修改建议生成最终的修改过后的文本。具体任务流程如下：
1. 参照树结构的文本总结，理解修改建议。树结构的文本总结提供了全文的修改建议，帮助你定位并修改当前段落。
2. 从树结构中找到与当前段落相关的部分，根据修改建议调整文本。
3. 返回修改过后的段落文本。
以下是原始文本的一段：
{text}

以下是全文的树结构总结：
{tree}

请根据树结构中的修改建议，调整当前段落的文本，你只需要提供修改过后的文本，不需要提供其他任何信息。"""


FINAL_MODIFY_CHUNK_EN = """You will modify a given paragraph based on a set of suggestions and a tree-structured summary of the full text. The process is as follows:
1. Refer to the tree-structured summary to understand the modification suggestions. The summary provides suggestions for the entire text, helping you locate and modify the relevant paragraph.
2. Find the relevant section in the tree structure that corresponds to the current paragraph and adjust the text according to the suggestions.
3. Return the modified version of the paragraph. 
Below is the original paragraph: 
{text}

Below is the full tree-structured summary: 
{tree}

Please adjust the current paragraph based on the suggestions in the tree structure. You only need to provide the modified paragraph text, without any additional information."""