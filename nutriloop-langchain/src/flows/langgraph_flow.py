from langchain import LLMChain
from langgraph import Graph

class LangGraphFlow:
    def __init__(self, llm_chain: LLMChain, graph: Graph):
        self.llm_chain = llm_chain
        self.graph = graph

    def run_flow(self, input_data):
        # Step 1: Process input data through the LLM chain
        llm_output = self.llm_chain.run(input_data)

        # Step 2: Use the output to update the graph
        self.graph.update(llm_output)

        # Step 3: Execute the graph to get results
        results = self.graph.execute()

        return results

    def add_node(self, node_data):
        self.graph.add_node(node_data)

    def add_edge(self, from_node, to_node):
        self.graph.add_edge(from_node, to_node)