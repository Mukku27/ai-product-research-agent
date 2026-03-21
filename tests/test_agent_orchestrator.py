import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "agent_orchestrator.py"


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"models": [{"name": "llama3"}]}


class FakeRequestsModule(types.ModuleType):
    def get(self, *args, **kwargs):
        return FakeResponse()


class FakeChain:
    def __or__(self, other):
        return self

    def invoke(self, inputs):
        return "hello"


class FakeChatPromptTemplate:
    @staticmethod
    def from_template(template):
        return FakeChain()


class FakeStrOutputParser:
    pass


class FakeBaseTool:
    pass


class FakeOllamaLLM:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeTask:
    def __init__(self, description, expected_output, agent, dependencies=None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.dependencies = dependencies or []


class FakeCrew:
    def __init__(self, agents, tasks, verbose, process, cache):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose
        self.process = process
        self.cache = cache
        self.kickoff_calls = []

    def kickoff(self, inputs):
        self.kickoff_calls.append(inputs)
        return types.SimpleNamespace(output="analysis output")


class FakeProcess:
    sequential = "sequential"


def load_agent_orchestrator():
    fake_crewai = types.ModuleType("crewai")
    fake_crewai.Agent = FakeAgent
    fake_crewai.Crew = FakeCrew
    fake_crewai.Process = FakeProcess
    fake_crewai.Task = FakeTask

    fake_crewai_tools = types.ModuleType("crewai.tools")
    fake_crewai_tools.BaseTool = FakeBaseTool

    fake_prompts = types.ModuleType("langchain_core.prompts")
    fake_prompts.ChatPromptTemplate = FakeChatPromptTemplate

    fake_output_parsers = types.ModuleType("langchain_core.output_parsers")
    fake_output_parsers.StrOutputParser = FakeStrOutputParser

    fake_ollama = types.ModuleType("langchain_ollama")
    fake_ollama.OllamaLLM = FakeOllamaLLM

    fake_requests = FakeRequestsModule("requests")

    fake_vector_db_client = types.ModuleType("vector_db_client")
    fake_vector_db_client.get_opensearch_client = lambda *args, **kwargs: object()

    fake_modules = {
        "crewai": fake_crewai,
        "crewai.tools": fake_crewai_tools,
        "langchain_core.prompts": fake_prompts,
        "langchain_core.output_parsers": fake_output_parsers,
        "langchain_ollama": fake_ollama,
        "requests": fake_requests,
        "vector_db_client": fake_vector_db_client,
    }

    originals = {name: sys.modules.get(name) for name in fake_modules}
    module_name = "agent_orchestrator_under_test"
    original_module = sys.modules.get(module_name)

    try:
        sys.modules.update(fake_modules)
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = original_module

        for name, original in originals.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class AgentOrchestratorTests(unittest.TestCase):
    def test_create_patent_analysis_crew_uses_supplied_research_area(self):
        module = load_agent_orchestrator()
        module.check_ollama_availability = lambda: ["llama3"]
        module.test_model = lambda model_name: True

        crew = module.create_patent_analysis_crew(
            model_name="llama3", research_area="solar panel recycling"
        )

        self.assertEqual(len(crew.tasks), 4)
        task_descriptions = [task.description.lower() for task in crew.tasks]
        task_outputs = [task.expected_output.lower() for task in crew.tasks]

        for description in task_descriptions:
            self.assertIn("solar panel recycling", description)
            self.assertNotIn("lithium battery", description)

        for expected_output in task_outputs:
            self.assertIn("solar panel recycling", expected_output)
            self.assertNotIn("lithium battery", expected_output)

    def test_run_patent_analysis_passes_normalized_research_area_to_crew(self):
        module = load_agent_orchestrator()
        captured = {}

        class RecordingCrew:
            def kickoff(self, inputs):
                captured["inputs"] = inputs
                return types.SimpleNamespace(output="analysis output")

        def fake_create_patent_analysis_crew(model_name, research_area):
            captured["model_name"] = model_name
            captured["research_area"] = research_area
            return RecordingCrew()

        module.create_patent_analysis_crew = fake_create_patent_analysis_crew

        result = module.run_patent_analysis("  solid state cooling  ", "llama3")

        self.assertEqual("llama3", captured["model_name"])
        self.assertEqual("solid state cooling", captured["research_area"])
        self.assertEqual(
            {"research_area": "solid state cooling"},
            captured["inputs"],
        )
        self.assertEqual("analysis output", result)

    def test_normalize_research_area_falls_back_to_default(self):
        module = load_agent_orchestrator()

        self.assertEqual(
            module.DEFAULT_RESEARCH_AREA,
            module.normalize_research_area("   "),
        )


if __name__ == "__main__":
    unittest.main()
