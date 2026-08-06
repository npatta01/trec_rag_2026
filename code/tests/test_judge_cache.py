"""Contract tests for the cross-run RAGDoll support-judge cache."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from trec_rag.judge_cache import (
    CACHE_SCHEMA_VERSION,
    JudgeCache,
    JudgeIdentity,
    JudgeCacheConflict,
)


def identity(**overrides: object) -> JudgeIdentity:
    base = JudgeIdentity(
        evaluator="support",
        instruction="Statement: alpha\nCitation: beta\n",
        ragdoll_version="0.1.0",
        ragdoll_commit="1" * 40,
        prompt_contract_sha256="2" * 64,
        task_schema_version="ragdoll_support_task_v1",
        provider="fixture-provider",
        model="fixture/model-1",
        thinking="medium",
        temperature=None,
        system_prompt="you are a judge",
        agent_binary="fixture-agent",
        extension_identity="none",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class CacheCase(unittest.TestCase):
    def cache(self) -> JudgeCache:
        directory = Path(tempfile.mkdtemp(prefix="judge-cache-test-"))
        self.addCleanup(__import__("shutil").rmtree, directory, True)
        return JudgeCache(directory)


class IdentityTests(CacheCase):
    def test_digest_ignores_provenance_only_values(self) -> None:
        """Different runs producing the same request must share a cache entry."""
        first = identity()
        second = identity()
        self.assertEqual(first.digest, second.digest)

    def test_digest_is_stable_across_processes(self) -> None:
        self.assertEqual(identity().digest, identity().digest)
        self.assertEqual(len(identity().digest), 64)

    def test_every_output_affecting_field_changes_the_digest(self) -> None:
        baseline = identity().digest
        variations = {
            "evaluator": "nugget",
            "instruction": "Statement: alpha\nCitation: gamma\n",
            "ragdoll_version": "0.2.0",
            "ragdoll_commit": "3" * 40,
            "prompt_contract_sha256": "4" * 64,
            "task_schema_version": "ragdoll_support_task_v2",
            "provider": "other-provider",
            "model": "other/model",
            "thinking": "high",
            "temperature": 0.5,
            "system_prompt": "different",
            "agent_binary": "other-agent",
            "extension_identity": "some-extension@1",
        }
        for field, value in variations.items():
            with self.subTest(field=field):
                self.assertNotEqual(identity(**{field: value}).digest, baseline)

    def test_identity_payload_carries_no_run_scoped_values(self) -> None:
        payload = json.dumps(identity().payload(), sort_keys=True)
        for forbidden in ("task_id", "topic_id", "run_id", "config", "work_dir", "timestamp"):
            self.assertNotIn(forbidden, payload)


class ReadWriteTests(CacheCase):
    def test_miss_then_hit(self) -> None:
        cache = self.cache()
        key = identity()
        self.assertIsNone(cache.get(key))
        cache.put(key, support_label="FS")
        self.assertEqual(cache.get(key), "FS")
        self.assertEqual(cache.stats()["hits"], 1)
        self.assertEqual(cache.stats()["misses"], 1)
        self.assertEqual(cache.stats()["writes"], 1)

    def test_a_second_cache_object_sees_the_entry(self) -> None:
        cache = self.cache()
        key = identity()
        cache.put(key, support_label="PS")
        self.assertEqual(JudgeCache(cache.root).get(key), "PS")

    def test_changed_instruction_misses(self) -> None:
        cache = self.cache()
        cache.put(identity(), support_label="FS")
        self.assertIsNone(cache.get(identity(instruction="Statement: alpha\nCitation: delta\n")))

    def test_invalid_label_is_refused(self) -> None:
        cache = self.cache()
        with self.assertRaises(ValueError):
            cache.put(identity(), support_label="MAYBE")

    def test_failures_are_never_cached(self) -> None:
        cache = self.cache()
        with self.assertRaises(ValueError):
            cache.put(identity(), support_label=None)  # type: ignore[arg-type]
        self.assertIsNone(cache.get(identity()))

    def test_conflicting_label_for_the_same_identity_is_rejected(self) -> None:
        cache = self.cache()
        cache.put(identity(), support_label="FS")
        with self.assertRaises(JudgeCacheConflict):
            cache.put(identity(), support_label="NS")
        self.assertEqual(cache.get(identity()), "FS")

    def test_rewriting_the_same_label_is_idempotent(self) -> None:
        cache = self.cache()
        cache.put(identity(), support_label="FS")
        cache.put(identity(), support_label="FS")
        self.assertEqual(cache.get(identity()), "FS")


class CorruptionTests(CacheCase):
    def corrupt(self, cache: JudgeCache, mutate) -> None:
        key = identity()
        cache.put(key, support_label="FS")
        path = cache.path_for(key.digest)
        stored = json.loads(path.read_text(encoding="utf-8"))
        mutate(stored)
        path.write_text(json.dumps(stored), encoding="utf-8")

    def assert_becomes_miss(self, mutate) -> None:
        cache = self.cache()
        self.corrupt(cache, mutate)
        self.assertIsNone(cache.get(identity()))
        self.assertEqual(cache.stats()["invalidations"], 1)

    def test_truncated_file_is_a_miss(self) -> None:
        cache = self.cache()
        key = identity()
        cache.put(key, support_label="FS")
        cache.path_for(key.digest).write_text("{not json", encoding="utf-8")
        self.assertIsNone(cache.get(key))
        self.assertEqual(cache.stats()["invalidations"], 1)

    def test_wrong_schema_version_is_a_miss(self) -> None:
        self.assert_becomes_miss(lambda stored: stored.update(schema_version="other"))

    def test_mismatched_digest_is_a_miss(self) -> None:
        self.assert_becomes_miss(lambda stored: stored.update(identity_sha256="0" * 64))

    def test_altered_identity_payload_is_a_miss(self) -> None:
        self.assert_becomes_miss(lambda stored: stored["identity"].update(model="swapped"))

    def test_incomplete_status_is_a_miss(self) -> None:
        self.assert_becomes_miss(lambda stored: stored.update(status="failed"))

    def test_invalid_stored_label_is_a_miss(self) -> None:
        self.assert_becomes_miss(lambda stored: stored.update(support_label="???"))

    def test_healed_entry_becomes_usable_again(self) -> None:
        cache = self.cache()
        self.corrupt(cache, lambda stored: stored.update(status="failed"))
        self.assertIsNone(cache.get(identity()))
        cache.put(identity(), support_label="PS")
        self.assertEqual(cache.get(identity()), "PS")

    def test_stored_entry_holds_no_raw_provider_output(self) -> None:
        cache = self.cache()
        cache.put(identity(), support_label="FS")
        stored = json.loads(cache.path_for(identity().digest).read_text(encoding="utf-8"))
        self.assertEqual(stored["schema_version"], CACHE_SCHEMA_VERSION)
        self.assertNotIn("raw_output", stored)
        self.assertNotIn("events", stored)


class AtomicityTests(CacheCase):
    def test_write_leaves_no_partial_files(self) -> None:
        cache = self.cache()
        cache.put(identity(), support_label="FS")
        leftovers = [path.name for path in cache.root.rglob("*") if path.is_file() and ".tmp" in path.name]
        self.assertEqual(leftovers, [])

    def test_entry_path_is_content_addressed(self) -> None:
        cache = self.cache()
        key = identity()
        cache.put(key, support_label="FS")
        self.assertTrue(cache.path_for(key.digest).is_file())
        self.assertIn(key.digest, cache.path_for(key.digest).name)


class PrivacyModeTests(CacheCase):
    def test_directories_and_files_are_owner_only(self) -> None:
        cache = self.cache()
        key = identity()
        cache.put(key, support_label="FS")
        path = cache.path_for(key.digest)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
        self.assertEqual(cache.root.stat().st_mode & 0o777, 0o700)

    def test_a_loose_root_is_tightened_on_open(self) -> None:
        directory = Path(tempfile.mkdtemp(prefix="judge-cache-loose-"))
        self.addCleanup(__import__("shutil").rmtree, directory, True)
        directory.chmod(0o755)
        JudgeCache(directory)
        self.assertEqual(directory.stat().st_mode & 0o777, 0o700)


class ConcurrencyTests(CacheCase):
    """Two writers must not race through a read-then-write window."""

    def test_conflicting_concurrent_writers_fail_closed(self) -> None:
        import threading

        cache = self.cache()
        key = identity()
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []
        labels = ["FS", "NS"]

        def writer(index: int) -> None:
            barrier.wait()
            try:
                JudgeCache(cache.root).put(key, support_label=labels[index])
            except BaseException as error:  # noqa: BLE001 - recorded for assertion
                errors.append(error)

        threads = [threading.Thread(target=writer, args=(index,)) for index in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(errors), 1, msg="exactly one writer must lose and fail closed")
        self.assertIsInstance(errors[0], JudgeCacheConflict)
        self.assertIn(cache.get(key), labels)

    def test_same_label_concurrent_writers_converge(self) -> None:
        import threading

        cache = self.cache()
        key = identity()
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def writer() -> None:
            barrier.wait()
            try:
                JudgeCache(cache.root).put(key, support_label="PS")
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(cache.get(key), "PS")


class PinnedSettingTests(unittest.TestCase):
    """Settings left out of the identity must be pinned, not silently variable."""

    def test_thinking_resolves_against_the_model_not_the_other_way_round(self) -> None:
        from ragdoll.config import DEFAULT_MODEL, DEFAULT_THINKING

        from trec_rag import competition_evaluation_report as cli

        pinned = cli.judge_settings(Path(__file__).parents[2])
        self.assertNotEqual(pinned.thinking, DEFAULT_MODEL)
        self.assertIn(pinned.thinking, {DEFAULT_THINKING, "minimal"})

    def test_excluded_agent_settings_are_fixed_by_the_production_config(self) -> None:
        from trec_rag import competition_evaluation_report as cli

        config = cli.local_agent_config(
            cli.judge_settings(Path(__file__).parents[2])
        )
        # Excluded because they cannot change a completed answer:
        self.assertIsNone(config.cache_dir)  # RAGDoll's TTL cache stays off
        self.assertIsNone(config.agent_state_dir)  # copied per call into a temp dir
        self.assertIsNone(config.extension_path)  # no extension is configured
        self.assertIsNone(config.extension_env)
        # And the identity records that "no extension" explicitly.
        self.assertEqual(cli.judge_settings(Path(__file__).parents[2]).extension_identity, "none")

    def test_identity_covers_every_output_affecting_agent_setting(self) -> None:
        from dataclasses import fields

        from ragdoll.config import LocalAgentConfig

        from trec_rag.judge_cache import JudgeIdentity

        agent_fields = {field.name for field in fields(LocalAgentConfig)}
        identity_fields = {field.name for field in fields(JudgeIdentity)}
        output_affecting = {
            "agent_binary",
            "provider",
            "model",
            "thinking",
            "system_prompt",
            "temperature",
        }
        self.assertTrue(output_affecting <= agent_fields)
        self.assertTrue(
            output_affecting <= identity_fields,
            msg=f"missing from the cache identity: {sorted(output_affecting - identity_fields)}",
        )
        # Everything else in the agent config is deliberately excluded and pinned.
        self.assertEqual(
            agent_fields - output_affecting,
            {"timeout_seconds", "agent_state_dir", "cache_dir", "extension_path",
             "extension_cwd", "extension_env"},
        )


if __name__ == "__main__":
    unittest.main()
