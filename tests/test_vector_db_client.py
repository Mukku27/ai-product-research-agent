import importlib
import sys
import types
import unittest


class FakeIndices:
    def __init__(self, exists):
        self._exists = exists
        self.deleted = []
        self.created = []

    def exists(self, index):
        return self._exists

    def delete(self, index):
        self.deleted.append(index)

    def create(self, index, body):
        self.created.append((index, body))


class FakeClient:
    def __init__(self, exists):
        self.indices = FakeIndices(exists=exists)


class CreateIndexIfNotExistsTests(unittest.TestCase):
    def setUp(self):
        self._modules = {}
        for name in ("opensearchpy", "semantic_encoder", "vector_db_client"):
            self._modules[name] = sys.modules.get(name)

        fake_opensearchpy = types.ModuleType("opensearchpy")
        fake_opensearchpy.OpenSearch = object
        sys.modules["opensearchpy"] = fake_opensearchpy

    def tearDown(self):
        for name, module in self._modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def load_module(self, embedding=None):
        fake_semantic_encoder = types.ModuleType("semantic_encoder")
        fake_semantic_encoder.get_embedding = embedding or (lambda _: [0.1, 0.2, 0.3])
        sys.modules["semantic_encoder"] = fake_semantic_encoder
        sys.modules.pop("vector_db_client", None)
        return importlib.import_module("vector_db_client")

    def test_existing_index_is_left_intact_by_default(self):
        vector_db_client = self.load_module()
        client = FakeClient(exists=True)

        created = vector_db_client.create_index_if_not_exists(client, "patents")

        self.assertFalse(created)
        self.assertEqual(client.indices.deleted, [])
        self.assertEqual(client.indices.created, [])

    def test_missing_index_is_created(self):
        vector_db_client = self.load_module(embedding=lambda _: [1.0, 2.0, 3.0, 4.0])
        client = FakeClient(exists=False)

        created = vector_db_client.create_index_if_not_exists(client, "patents")

        self.assertTrue(created)
        self.assertEqual(client.indices.deleted, [])
        self.assertEqual(len(client.indices.created), 1)
        _, mappings = client.indices.created[0]
        self.assertEqual(
            mappings["mappings"]["properties"]["embedding"]["dimension"],
            4,
        )

    def test_recreate_flag_makes_deletion_explicit(self):
        vector_db_client = self.load_module(embedding=lambda _: [0.5, 0.6])
        client = FakeClient(exists=True)

        created = vector_db_client.create_index_if_not_exists(
            client,
            "patents",
            recreate=True,
        )

        self.assertTrue(created)
        self.assertEqual(client.indices.deleted, ["patents"])
        self.assertEqual(len(client.indices.created), 1)


if __name__ == "__main__":
    unittest.main()
