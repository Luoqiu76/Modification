import asyncio
from typing import List, TypedDict, Union
from llm import LLM
from utils import *

class Node:
    # if parent is None, then it is the root node
    # if children is [], then it is a leaf node
    def __init__(self, text: str, children : List['Node'], parent : Union['Node', None], start : str, end : str, summary : str):
        self.text = text
        self.children = children
        self.parent = parent
        self.start = start
        self.end = end
        self.summary = summary
    def __str__(self):
        return str(
            {
                "from" : self.start,
                "to" : self.end,
                "summary" : self.summary,
            }
        )
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


class Entity(TypedDict):
    entity : str
    modification : str
    importance : int


class GetEntities:
    def __init__(self, prompts: dict, llm: LLM, llm_kwargs: dict):
        self.prompts = prompts
        self.llm = llm
        self.llm_kwargs = llm_kwargs

    def forward_sync(self, text: str) -> List[Entity]:
        prompt = self.prompts['extract_entities'].format(
            content = text
        )
        response = self.llm.get_response_sync(
            prompt = prompt,
            log_stage = "Extract Entities",
            **self.llm_kwargs
        )
        entities = [Entity(**entity) for entity in process_json_output(response)]
        return entities
    
    async def forward_async(self, text: str) -> List[Entity]:
        prompt = self.prompts['extract_entities'].format(
            content = text
        )
        response = await self.llm.get_response_async(
            prompt = prompt,
            log_stage = "Extract Entities",
            **self.llm_kwargs
        )
        entities = [Entity(**entity) for entity in process_json_output(response)]
        return entities


class GetLevelNodes:
    def __init__(self, prompts: dict, llm: LLM, depth_limit: int, llm_kwargs: dict):
        self.prompts = prompts
        self.llm = llm
        self.llm_kwargs = llm_kwargs
        self.depth_limit = depth_limit
    # based on text and the father node, construct new nodes, the text is the content of the father node, the depth is not the depth of the father node, but the depth of the node need to be constructed
    def forward_sync(self, text: str, level: int, father_node: Union[Node, None] = None, entities: List[str] = []) -> List[Node]:
        ## Stop condition 1: the level is greater than the depth limit
        if level > self.depth_limit:
            return []
        prompt = self.prompts['get_level_nodes'].format(
            text = text,
            entities = str(entities)
        )
        response = self.llm.get_response_sync(
            prompt = prompt,
            log_stage = f"Get Level Nodes in Level {level}",
            **self.llm_kwargs
        )

        ## Stop condition 2: the response is DONE
        if response.strip().upper() == 'DONE':
            return []

        nodes = [Node(
            text = "",
            children = [],
            parent = father_node,
            **node
        ) for node in process_json_output(response)]

        def judge_is_leaf_and_set_text(node: Node) -> bool:
            prompt = self.prompts['find'].format(
                text = text, # text is the argument of the function, which is the content of the father node
                start = node['start'],
                end = node['end']
            )
            response = self.llm.get_response_sync(
                prompt = prompt,
                log_stage = f"Find Node Start and End in Level {level}",
                **self.llm_kwargs
            )
            # if the response is nearly same as the text, then it is a leaf node
            if min(len(response), len(text)) / max(len(response), len(text)) > 0.9:
                return True
            else:
                node['text'] = response
                return False
        judge_results = [judge_is_leaf_and_set_text(node) for node in nodes]
        ## Stop condition 3: any node is a leaf node
        if any(judge_results):
            return []


        for node in nodes:
            node["children"] = self.forward_sync(
                text = node["text"],
                level = level + 1,
                father_node = node,
                entities = entities
            )
        return nodes
    

    async def forward_async(self, text: str, level: int, father_node: Union[Node, None] = None, entities: List[str] = []) -> List[Node]:
        ## Stop condition 1: the level is greater than the depth limit
        if level > self.depth_limit:
            return []
        prompt = self.prompts['get_level_nodes'].format(
            text = text,
            entities = str(entities)
        )
        response = await self.llm.get_response_async(
            prompt = prompt,
            log_stage = f"Get Level Nodes in Level {level}",
            **self.llm_kwargs
        )
        # print(prompt)
        # print("=====================================")
        # print(response)
        # print("=====================================")
        ## Stop condition 2: the response is DONE
        if response.strip().upper() == 'DONE':
            return []

        nodes = [Node(
            text = "",
            children = [],
            parent = father_node,
            **node
        ) for node in process_json_output(response)]

        async def judge_is_leaf_and_set_text(node: Node) -> bool:
            prompt = self.prompts['find'].format(
                text = text, # text is the argument of the function, which is the content of the father node
                start = node['start'],
                end = node['end']
            )
            response = await self.llm.get_response_async(
                prompt = prompt,
                log_stage = f"Find Node Start and End in Level {level}",
                **self.llm_kwargs
            )
            # if the response is nearly same as the text, then it is a leaf node
            if min(len(response), len(text)) / max(len(response), len(text)) > 0.9:
                return True
            else:
                node['text'] = response
                return False
        judge_results = [await judge_is_leaf_and_set_text(node) for node in nodes]
        ## Stop condition 3: any node is a leaf node
        if any(judge_results):
            return []
        # gather is in the order of the list, so we need to use asyncio.gather to keep the order of the nodes if use as_completed, the order will be changed

        node_children = await asyncio.gather(*[self.forward_async(
            text = node["text"],
            level = level + 1,
            father_node = node,
            entities = entities
        ) for node in nodes])

        for i, node in enumerate(nodes):
            node["children"] = node_children[i]
        return nodes


class GetTreebasedModification:
    def __init__(self, prompts: dict, llm: LLM, llm_kwargs: dict):
        self.prompts = prompts
        self.llm = llm
        self.llm_kwargs = llm_kwargs

    def get_tree_json(self, root_node: Node) -> dict:
        def dfs(root: Node, level: int = 0) -> dict:
            return {
                "level": level,
                "start" : root['start'],
                "end" : root['end'],
                "summary" : root['summary'],
                "children" : [dfs(child, level + 1) for child in root['children']]
            }
        return dfs(root_node)

    def forward_sync(self, root_node: Node, overall_modification: str) -> str:
        prompt = self.prompts['base_tree_modification'].format(
            tree = self.get_tree_json(root_node),
            modification = overall_modification
        )
        response = self.llm.get_response_sync(
            prompt = prompt,
            log_stage = "Get Treebased Modification",
            **self.llm_kwargs
        )
        return process_json_output(response)
    
    async def forward_async(self, root_node: Node, overall_modification: str) -> dict:
        prompt = self.prompts['base_tree_modification'].format(
            tree = self.get_tree_json(root_node),
            modification = overall_modification
        )
        response = await self.llm.get_response_async(
            prompt = prompt,
            log_stage = "Get Treebased Modification",
            **self.llm_kwargs
        )
        response = process_json_output(response)
        return response

class FinalModification:
    def __init__(self, prompts: dict, llm: LLM, llm_kwargs: dict, chunks_config : dict):
        self.prompts = prompts
        self.llm = llm
        self.llm_kwargs = llm_kwargs
        self.chunks_config = chunks_config

    def forward_sync(self, text: str, tree_modification: dict) -> str:
        if not get_config("pipeline_config")['enable_response_chunking']:
            prompt = self.prompts['final_modification'].format(
                text = text,
                tree = tree_modification
            )
            response = self.llm.get_response_sync(
                prompt = prompt,
                log_stage = "Get Final Modification",
                **self.llm_kwargs
            )
            return response
        else:
            text_spilter = FixedTokensTextSplitter(
                **self.chunks_config
            )
            chunks = text_spilter.get_chunks(text)
            responses = []
            for chunk in chunks:
                prompt = self.prompts['final_modification_naive_chunk'].format(
                    text = chunk,
                    tree = tree_modification
                )
                response = self.llm.get_response_sync(
                    prompt = prompt,
                    log_stage = "Get Final Modification",
                    **self.llm_kwargs
                )
                responses.append(response)
            return "".join(responses)
    async def forward_async(self, text: str, tree_modification: dict) -> str:
        if not get_config("pipeline_config")['enable_response_chunking']:
            prompt = self.prompts['final_modification'].format(
                text = text,
                tree = tree_modification
            )
            response = await self.llm.get_response_async(
                prompt = prompt,
                log_stage = "Get Final Modification",
                **self.llm_kwargs
            )
            return response
        else:
            text_spilter = FixedTokensTextSplitter(
                **self.chunks_config
            )
            chunks = text_spilter.get_chunks(text)
            responses = await asyncio.gather(*[self.llm.get_response_async(
                    prompt = self.prompts['final_modification_naive_chunk'].format(
                        text = chunk,
                        tree = tree_modification
                    ),
                    log_stage = "Get Final Modification",
                    **self.llm_kwargs
                )
                for chunk in chunks])
            return "".join(responses)
        
class FinalModificationNaiveChunk(FinalModification):
    def __init__(self, prompts: dict, llm: LLM, llm_kwargs: dict, chunks_config : dict):
        super().__init__(prompts, llm, llm_kwargs, chunks_config)
    
    def forward_sync(self, text: str, tree_modification: dict) -> str:
        text_spilter = FixedTokensTextSplitter(
            **self.chunks_config 
        )
        chunks = text_spilter.get_chunks(text)
        responses = []
        for chunk in chunks:
            prompt = self.prompts['final_modification_naive_chunk'].format(
                text = chunk,
                tree = tree_modification
            )
            response = self.llm.get_response_sync(
                prompt = prompt,
                log_stage = "Get Final Modification",
                **self.llm_kwargs
      
            )
            responses.append(response)
        return "".join(responses)

    async def forward_async(self, text: str, tree_modification: dict) -> str:
        text_spilter = FixedTokensTextSplitter(
            **self.chunks_config 
        )
        chunks = text_spilter.get_chunks(text)
        responses = await asyncio.gather(*[self.llm.get_response_async(
                prompt = self.prompts['final_modification_naive_chunk'].format(
                    text = chunk,
                    tree = tree_modification
                ),
                log_stage = "Get Final Modification",
                **self.llm_kwargs
            )
            for chunk in chunks])
        return "".join(responses)
    


class FinalModificationContextChunk(FinalModification):
    def __init__(self, prompts: dict, llm: LLM, llm_kwargs: dict, chunks_config : dict):
        super().__init__(prompts, llm, llm_kwargs, chunks_config)

    def forward_sync(self, text: str, tree_modification: dict) -> str:
        context_chunk_config = get_config("pipeline_config/context_chunk_config")
        context_ratio = context_chunk_config['context_ratio']
        text_spilter_chunk_length = int(self.chunks_config['chunk_size'] * (1 - context_ratio))
        self.chunks_config['chunk_size'] = text_spilter_chunk_length
        text_spilter = FixedTokensTextSplitter(
            **self.chunks_config 
        )
        chunks = text_spilter.get_chunks(text)
        responses = []
        for chunk in chunks:
            previos_text = responses[-1] if len(responses) > 0 else ""
            previos_text = previos_text[-int(self.chunks_config['chunk_size'] * context_ratio):] if len(previos_text) > 0 else ""
            prompt = self.prompts['final_modification_context_chunk'].format(
                text = chunk,
                tree = tree_modification,
                previous_text = previos_text
            )
            response = self.llm.get_response_sync(
                prompt = prompt,
                log_stage = "Get Final Modification",
                **self.llm_kwargs
            )
            responses.append(response)
        return "".join(responses)
    async def forward_async(self, text: str, tree_modification: dict) -> str:
        context_chunk_config = get_config("pipeline_config/context_chunk_config")
        context_ratio = context_chunk_config['context_ratio']
        text_spilter_chunk_length = int(self.chunks_config['chunk_size'] * (1 - context_ratio))
        self.chunks_config['chunk_size'] = text_spilter_chunk_length
        text_spilter = FixedTokensTextSplitter(
            **self.chunks_config 
        )
        chunks = text_spilter.get_chunks(text)
        responses = []
        for chunk in chunks:
            previos_text = responses[-1] if len(responses) > 0 else ""
            previos_text = previos_text[-int(self.chunks_config['chunk_size'] * context_ratio):] if len(previos_text) > 0 else ""
            prompt = self.prompts['final_modification_context_chunk'].format(
                text = chunk,
                tree = tree_modification,
                previous_text = previos_text
            )
            response = await self.llm.get_response_async(
                prompt = prompt,
                log_stage = "Get Final Modification",
                **self.llm_kwargs
            )
            responses.append(response)
        return "".join(responses)

class Pipeline:
    def __init__(self, prompts: dict, llm: LLM):
        self.prompts = prompts
        self.llm = llm
        self.llm_kwargs = get_config("model_config")
        self.get_entities = GetEntities(
            prompts = prompts,
            llm = self.llm,
            llm_kwargs = self.llm_kwargs
        )
        self.get_level_nodes = GetLevelNodes(
            prompts = prompts,
            llm = self.llm,
            depth_limit = get_config("pipeline_config")['depth_limit'],
            llm_kwargs = self.llm_kwargs
        )
        self.get_treebased_modification = GetTreebasedModification(
            prompts = prompts,
            llm = self.llm,
            llm_kwargs = self.llm_kwargs
        )
        if get_config("pipeline_config")['enable_response_chunking']:
            self.get_final_modification = FinalModificationNaiveChunk(
            prompts = prompts,
            llm = self.llm,
            llm_kwargs = self.llm_kwargs,
            chunks_config = get_config('pipeline_config/chunks_config')
        )
        elif get_config("pipeline_config")['enable_response_context_chunking']:
            self.get_final_modification = FinalModificationContextChunk(
            prompts = prompts,
            llm = self.llm,
            llm_kwargs = self.llm_kwargs,
            chunks_config = get_config('pipeline_config/chunks_config')
        )
        else:
            self.get_final_modification = FinalModification(
                prompts = prompts,
                llm = self.llm,
                llm_kwargs = self.llm_kwargs,
                chunks_config = get_config('pipeline_config/chunks_config')
            )
    @retry_on_failure_sync(max_retries=get_config()['max_retries'], return_time=True)   
    def forward_sync(self, text: str, overall_modification: str) -> str:
        entities = self.get_entities.forward_sync(overall_modification) # List[Entity]
        root_node = Node(
            text = text,
            children = [],
            parent = None,
            start = text[:15],
            end = text[-15:],
            summary = "The whole text",
        )
        root_node_children = []
        entities = [entity['entity'] for entity in entities]
        if get_config("pipeline_config/chunks_config")['chunk_size'] == "NO_LIMIT":
            root_node_children.extend(
                    self.get_level_nodes.forward_sync(
                    text = text,
                    level = 1,
                    father_node = root_node,
                    entities = entities
                )
            )
        else:
            text_spilter = FixedTokensTextSplitter(
                **get_config("pipeline_config/chunks_config")
            )
            chunks = text_spilter.get_chunks(text)
            for chunk in chunks:
                root_node_children.extend(
                        self.get_level_nodes.forward_sync(
                        text = chunk,
                        level = 1,
                        father_node = root_node,
                        entities = entities
                    )
                )
        root_node.children = root_node_children
        tree_based_modification = self.get_treebased_modification.forward_sync(
            root_node = root_node,
            overall_modification = overall_modification
        )
        return self.get_final_modification.forward_sync(
            text = text,
            tree_modification = tree_based_modification
        )
    @retry_on_failure_async(max_retries=get_config()['max_retries'], max_concurrent=get_config('async_config')['max_concurrent'], return_time=True)
    async def forward_async(self, text: str, overall_modification: str) -> str:
        entities = await self.get_entities.forward_async(overall_modification) # List[Entity]
        root_node = Node(
            text = text,
            children = [],
            parent = None,
            start = text[:15],
            end = text[-15:],
            summary = "The whole text",
        )
        root_node_children = []
        entities = [entity['entity'] for entity in entities]
        if get_config("pipeline_config/chunks_config")['chunk_size'] == "NO_LIMIT":
            root_node_children.extend(
                    await self.get_level_nodes.forward_async(
                    text = text,
                    level = 1,
                    father_node = root_node,
                    entities = entities
                )
            )
        else:
            text_spilter = FixedTokensTextSplitter(
                **get_config("pipeline_config/chunks_config")
            )
            chunks = text_spilter.get_chunks(text)
            tasks = [asyncio.ensure_future(
                    self.get_level_nodes.forward_async(
                    text = chunk,
                    level = 1,
                    father_node = root_node,
                    entities = entities
                )
            )
            for chunk in chunks]
            task_results = await asyncio.gather(*tasks)
            for result in task_results:
                root_node_children.extend(result)
        root_node.children = root_node_children
        await self.get_treebased_modification.forward_async(
            root_node = root_node,
            overall_modification = overall_modification
        )
        return await self.get_final_modification.forward_async(
            text = text,
            tree_modification = self.get_treebased_modification.get_tree_json(root_node)
        )


        



    
        

