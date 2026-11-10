"""Verify community governance, conduct, and contributor quickstart documentation (`COMM-01`, `COMM-02`, `COMM-03`)."""
from __future__ import annotations

from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CommunityDocsTests(unittest.TestCase):
    def test_code_of_conduct_exists_and_covers_reporting_and_scope(self):
        path = ROOT / "CODE_OF_CONDUCT.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Expected Behavior", text)
        self.assertIn("Unacceptable Behavior", text)
        self.assertIn("Reporting Conduct or Safety Concerns", text)
        self.assertIn("Enforcement Ladder", text)
        self.assertIn("does **not** claim a staffed 24/7 trust-and-safety team", text)

    def test_governance_roster_lists_workstreams_and_opt_in_vacancies(self):
        path = ROOT / "GOVERNANCE.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        for ws in ("OVERALL", "BASE", "EVAL", "ARCH", "DATA", "DIST", "COMM"):
            self.assertIn(f"**`{ws}`**", text)
        self.assertIn("VACANT (Opt-In Open)", text)
        self.assertIn("explicit opt-in consent", text)

    def test_contributing_quickstart_covers_five_offline_paths(self):
        path = ROOT / "CONTRIBUTING.md"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Zero-Credential Offline Quickstart", text)
        self.assertIn("requirements-aix-zero.txt", text)
        self.assertIn("Path A: Code & Architecture", text)
        self.assertIn("Path B: Evaluation, Verifiers & Statistical Gates", text)
        self.assertIn("Path C: Datasets & Provenance", text)
        self.assertIn("Path D: Independent Reproduction & Documentation", text)
        self.assertIn("Path E: Opt-In Compute Contributions", text)
        self.assertIn("Claim Proposal (`<TASK-CODE>`)", text)
        self.assertIn("No Unconsented Assignment or Review Promises", text)

    def test_license_and_provenance_policy_are_present(self):
        lic_path = ROOT / "LICENSE"
        pol_path = ROOT / "docs/LICENSE-AND-PROVENANCE-POLICY.md"
        self.assertTrue(lic_path.is_file())
        self.assertTrue(pol_path.is_file())
        self.assertIn("Apache License", lic_path.read_text(encoding="utf-8"))
        pol_text = pol_path.read_text(encoding="utf-8")
        self.assertIn("Apache-2.0", pol_text)
        self.assertIn("Explicit Exceptions: Datasets, Web Excerpts, and Checkpoints", pol_text)

    def test_wave3_replication_triage_reasoning_website_and_pr_templates(self):
        rep_doc = ROOT / "docs/REPLICATION-AND-TRIAGE.md"
        reason_doc = ROOT / "docs/REASONING-CAPABILITY-SPEC.md"
        web_doc = ROOT / "docs/WEBSITE-INFORMATION-ARCHITECTURE.md"
        pr_default = ROOT / ".github/pull_request_template.md"
        pr_exp = ROOT / ".github/PULL_REQUEST_TEMPLATE/experiment_or_research.md"
        pr_code = ROOT / ".github/PULL_REQUEST_TEMPLATE/code_or_docs.md"

        for p in (rep_doc, reason_doc, web_doc, pr_default, pr_exp, pr_code):
            self.assertTrue(p.is_file(), f"Missing {p}")

        self.assertIn("Independent Reproducer", rep_doc.read_text(encoding="utf-8"))
        self.assertIn("NON_IDENTIFIABLE", reason_doc.read_text(encoding="utf-8"))
        self.assertIn("Distributed coordinator: OFFLINE", web_doc.read_text(encoding="utf-8"))
        self.assertIn("Neural-Only Exact Match", pr_exp.read_text(encoding="utf-8"))

    def test_wave4_web_02_03_and_comm_08_10_11(self):
        from aix_zero.website_data_builder import build_project_overview_payload, build_roadmap_view_payload

        overview = build_project_overview_payload()
        self.assertEqual(overview["measured_capabilities"]["physical_learned_parameters"], 817410)
        self.assertEqual(overview["measured_capabilities"]["neural_only_accuracy"]["correct"], 9)
        self.assertEqual(overview["measured_capabilities"]["symbolic_assisted_accuracy"]["correct"], 105)
        self.assertEqual(overview["rejected_and_negative_experiments"][0]["status"], "REJECTED")

        # WEB-03 roadmap view & stale response handling
        fresh_view = build_roadmap_view_payload(
            [
                {"number": 32, "title": "[AIX BASE-09] Recovery", "state": "OPEN", "labels": ["status:ready"]},
                {"number": 24, "title": "[AIX BASE-01] Paths", "state": "CLOSED", "labels": []},
            ],
            fetched_at_utc="2026-11-10T22:00:00Z",
            now_utc="2026-11-10T22:10:00Z",
        )
        self.assertFalse(fresh_view["is_stale"])
        self.assertEqual(fresh_view["data_status"], "LIVE_FRESH")
        self.assertEqual(len(fresh_view["ready_to_claim"]), 1)

        stale_view = build_roadmap_view_payload(
            None,
            fetched_at_utc="2026-11-10T10:00:00Z",
            now_utc="2026-11-10T22:10:00Z",
        )
        self.assertTrue(stale_view["is_stale"])
        self.assertEqual(stale_view["data_status"], "UNAVAILABLE_FALLBACK")

        comm_doc = ROOT / "docs/COMMUNITY-TEMPLATES-AND-ONBOARDING.md"
        self.assertTrue(comm_doc.is_file())
        comm_text = comm_doc.read_text(encoding="utf-8")
        self.assertIn("Measured Progress-Update Template (`COMM-08`", comm_text)
        self.assertIn("Evidence-Based Recruitment Announcement (`COMM-10`", comm_text)
        self.assertIn("Three Well-Scoped Beginner Contribution Paths (`COMM-11`", comm_text)

    def test_wave5_worker_01_sec_01_network_02_and_web_08_11(self):
        import tempfile
        from aix_zero.checkpoint_registry import make_network_job_contract
        from aix_zero.network_coordinator import (
            DurableExperimentQueue,
            THREAT_BOUNDARY_MATRIX,
            match_job_to_worker,
            validate_worker_capability_handshake,
        )
        from aix_zero.website_data_builder import (
            audit_html_accessibility_and_keyboard_nav,
            build_research_updates_feed,
        )

        # WORKER-01 capability handshake & PII rejection
        worker_caps = {
            "worker_ephemeral_id": "worker-anon-01",
            "device_types": ["cpu"],
            "available_ram_mb": 2048,
            "available_disk_mb": 4096,
            "torch_version": "2.2.2",
            "safetensors_version": "0.8.0",
            "max_lease_seconds": 600,
            "allowed_job_types": ["OFFLINE_EVALUATION_JOB"],
        }
        validate_worker_capability_handshake(worker_caps)
        with self.assertRaisesRegex(ValueError, "prohibited identity/PII fields"):
            validate_worker_capability_handshake(dict(worker_caps, hostname="secret-laptop.local"))

        job = make_network_job_contract(
            {
                "schema_version": "1.0.0",
                "job_id": "job-eval-v11-002",
                "job_type": "OFFLINE_EVALUATION_JOB",
                "experiment_config_sha256": "a" * 64,
                "inputs": {
                    "base_checkpoint_path": "aix_zero/checkpoints/aix_zero_v11_kip.safetensors",
                    "base_checkpoint_sha256": "b" * 64,
                    "dataset_id": "v11-kip-test",
                    "dataset_sha256": "c" * 64,
                    "seed": 42,
                    "target_expert_slot": None,
                },
                "resource_budget": {
                    "max_steps": 0,
                    "max_wall_clock_seconds": 120,
                    "max_memory_mb": 512,
                    "device": "cpu",
                },
                "lease": {
                    "lease_id": "lease-002",
                    "worker_id": "worker-anon-01",
                    "issued_at_utc": "2026-11-10T22:00:00Z",
                    "expires_at_utc": "2026-11-10T22:30:00Z",
                    "heartbeat_interval_seconds": 15,
                    "attempt_number": 1,
                },
                "expected_outputs": {
                    "requires_evaluation_result_json": True,
                    "requires_candidate_safetensors": False,
                    "requires_failure_record_on_abort": True,
                },
            }
        )
        matched = match_job_to_worker(job, worker_caps)
        self.assertTrue(matched["matched"])
        low_ram_worker = dict(worker_caps, available_ram_mb=128)
        unmatched = match_job_to_worker(job, low_ram_worker)
        self.assertFalse(unmatched["matched"])
        self.assertEqual(unmatched["response_code"], "NO_WORK_UNSUPPORTED_CAPABILITY")

        # SEC-01 threat boundaries & NETWORK-02 durable queue restart persistence
        self.assertGreaterEqual(len(THREAT_BOUNDARY_MATRIX), 3)
        self.assertTrue((ROOT / "docs/WORKER-AND-COORDINATOR-THREAT-MODEL.md").is_file())

        with tempfile.TemporaryDirectory() as tmp:
            q_path = Path(tmp) / "queue.json"
            q1 = DurableExperimentQueue(q_path)
            q1.enqueue(job)
            q1.transition("job-eval-v11-002", "LEASED", worker_id="worker-anon-01")
            # Simulate coordinator restart by instantiating a fresh queue from q_path
            q2 = DurableExperimentQueue(q_path)
            self.assertEqual(q2.get_job("job-eval-v11-002")["status"], "LEASED")
            q2.transition("job-eval-v11-002", "COMPLETED", worker_id="worker-anon-01", result_summary={"ok": True})
            with self.assertRaisesRegex(ValueError, "Invalid job state transition"):
                q2.transition("job-eval-v11-002", "QUEUED")

        # WEB-08 updates feed & WEB-11 accessibility audit
        feed = build_research_updates_feed(
            [
                {
                    "update_id": "2026-11-10-m0-audit",
                    "date_utc": "2026-11-10T22:00:00Z",
                    "completed_work": ["Completed M0 foundation readiness audit (BASE-16)"],
                    "measurements": [
                        {
                            "metric": "v11 neural_only exact match",
                            "value": "9/105 (8.57%)",
                            "claims_capability_improvement": False,
                            "evidence_path": "research/baseline/v11-local.json",
                        }
                    ],
                    "hypotheses": [{"title": "Multi-layer adapter may overcome pointer copy bias"}],
                    "next_experiments": ["Validation-only adapter ablation on BP/BS"],
                    "corrections": ["Clarified v11 latency is warm-cache local CPU only"],
                }
            ]
        )
        self.assertEqual(feed["update_count"], 1)
        self.assertEqual(feed["updates"][0]["hypotheses"][0]["verification_status"], "UNVERIFIED_HYPOTHESIS")

        sample_html = """<!doctype html><html lang="en"><head><style>:focus-visible{outline:2px solid #005fcc}@media (prefers-reduced-motion:reduce){*{transition:none}}</style></head><body><a href="#main">Skip to main</a><nav aria-label="Primary">Nav</nav><main id="main"><h1>AIX-Zero</h1></main></body></html>"""
        a11y = audit_html_accessibility_and_keyboard_nav(sample_html)
        self.assertTrue(a11y["automated_passed"])

    def test_wave6_worker_02_13_network_03_and_sec_02_03_06_08(self):
        import hashlib
        import tempfile
        from aix_zero.checkpoint_registry import make_network_job_contract
        from aix_zero.network_coordinator import (
            DurableExperimentQueue,
            execute_allowlisted_local_job,
            probe_worker_accelerator_capabilities,
            sanitize_public_telemetry_and_logs,
            sign_release_manifest,
            validate_remote_artifact_stream,
            verify_release_manifest_signature,
            verify_worker_sandbox_isolation,
        )

        job = make_network_job_contract(
            {
                "schema_version": "1.0.0",
                "job_id": "job-eval-v11-003",
                "job_type": "OFFLINE_EVALUATION_JOB",
                "experiment_config_sha256": "a" * 64,
                "inputs": {
                    "base_checkpoint_path": "aix_zero/checkpoints/aix_zero_v11_kip.safetensors",
                    "base_checkpoint_sha256": "b" * 64,
                    "dataset_id": "v11-kip-test",
                    "dataset_sha256": "c" * 64,
                    "seed": 42,
                    "target_expert_slot": None,
                },
                "resource_budget": {
                    "max_steps": 0,
                    "max_wall_clock_seconds": 120,
                    "max_memory_mb": 512,
                    "device": "cpu",
                },
                "lease": {
                    "lease_id": "lease-003",
                    "worker_id": "worker-anon-01",
                    "issued_at_utc": "2026-11-10T22:00:00Z",
                    "expires_at_utc": "2026-11-10T22:30:00Z",
                    "heartbeat_interval_seconds": 15,
                    "attempt_number": 1,
                },
                "expected_outputs": {
                    "requires_evaluation_result_json": True,
                    "requires_candidate_safetensors": False,
                    "requires_failure_record_on_abort": True,
                },
            }
        )

        # WORKER-02 allowlisted local job runner
        run_res = execute_allowlisted_local_job(
            job,
            workspace_root=ROOT,
            allowed_runners={"OFFLINE_EVALUATION_JOB": lambda inp: {"evaluated_seed": inp["seed"]}},
        )
        self.assertTrue(run_res["executed"])

        # NETWORK-03 leases and idempotent job completion (preventing duplicate credit or split-brain winners)
        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            q = DurableExperimentQueue(tdir / "q.json")
            q.enqueue(job)
            q.acquire_lease("job-eval-v11-003", worker_id="w1", lease_id="L1", expires_at_epoch=100, now_epoch=50)
            # Lease L1 expires at t=100; w2 acquires lease L2 at t=110
            q.acquire_lease("job-eval-v11-003", worker_id="w2", lease_id="L2", expires_at_epoch=200, now_epoch=110)

            # Stale w1 tries to complete L1 -> rejected!
            stale_res = q.complete_lease_idempotent(
                "job-eval-v11-003", lease_id="L1", worker_id="w1", now_epoch=120, result_summary={"score": 1.0}
            )
            self.assertEqual(stale_res["outcome"], "REJECTED_STALE_OR_SUPERSEDED_LEASE")
            self.assertFalse(stale_res["credit_granted"])

            # Active w2 completes L2 -> accepted!
            canon_res = q.complete_lease_idempotent(
                "job-eval-v11-003", lease_id="L2", worker_id="w2", now_epoch=125, result_summary={"score": 1.0}
            )
            self.assertEqual(canon_res["outcome"], "ACCEPTED_CANONICAL_COMPLETION")
            self.assertTrue(canon_res["credit_granted"])

            # Duplicate upload of L2 -> idempotent no-op without duplicate credit!
            dup_res = q.complete_lease_idempotent(
                "job-eval-v11-003", lease_id="L2", worker_id="w2", now_epoch=126, result_summary={"score": 1.0}
            )
            self.assertEqual(dup_res["outcome"], "DUPLICATE_IGNORED_IDEMPOTENT")
            self.assertFalse(dup_res["credit_granted"])

            # SEC-03 remote artifact size, hash, and path traversal guard
            payload = b"safetensors-mock-bytes"
            sha = hashlib.sha256(payload).hexdigest()
            self.assertTrue(
                validate_remote_artifact_stream(payload, filename="shard.safetensors", target_dir=tdir, expected_sha256=sha)["verified"]
            )
            with self.assertRaisesRegex(ValueError, "Path traversal rejected"):
                validate_remote_artifact_stream(payload, filename="../escape.safetensors", target_dir=tdir, expected_sha256=sha)

            # SEC-02 sandbox isolation check
            sb_ok = verify_worker_sandbox_isolation(tdir, attempted_path=tdir / "sub/file.json", allow_network=False)
            self.assertTrue(sb_ok["sandbox_compliant"])
            sb_escape = verify_worker_sandbox_isolation(tdir, attempted_path=Path("/etc/passwd"), allow_network=False)
            self.assertFalse(sb_escape["sandbox_compliant"])

        # SEC-08 release provenance, key rotation, and rollback protection
        keyring = {
            "key-2026-old": {"status": "ROTATED_REVOKED", "secret_key": "old-secret"},
            "key-2026-active": {"status": "ACTIVE", "secret_key": "active-secret"},
        }
        signed = sign_release_manifest(
            {"release_version": "1.2.0", "package_sha256": "a" * 64},
            key_id="key-2026-active",
            secret_key="active-secret",
        )
        self.assertTrue(
            verify_release_manifest_signature(signed, trust_root_keyring=keyring, minimum_compatible_version=(1, 1, 0))["verified"]
        )
        with self.assertRaisesRegex(ValueError, "below minimum_compatible_version"):
            verify_release_manifest_signature(signed, trust_root_keyring=keyring, minimum_compatible_version=(2, 0, 0))

        # WORKER-13 GPU/accelerator capability probes
        probe_ok = probe_worker_accelerator_capabilities(requested_device="cpu")
        self.assertTrue(probe_ok["eligible_for_job_reservation"])
        probe_bad = probe_worker_accelerator_capabilities(requested_device="cpu", inject_nonfinite_fault=True)
        self.assertFalse(probe_bad["eligible_for_job_reservation"])

        # SEC-06 log minimization & redaction
        sanitized = sanitize_public_telemetry_and_logs(
            {
                "worker_ephemeral_id": "w-01",
                "hostname": "secret-host",
                "api_key": "sk-secret",
                "log_message": "Loaded /Users/alice/private/ckpt from 192.168.1.50",
            }
        )
        self.assertNotIn("hostname", sanitized)
        self.assertNotIn("api_key", sanitized)
        self.assertIn("<REDACTED_HOME_PATH>", sanitized["log_message"])
        self.assertIn("<REDACTED_IP>", sanitized["log_message"])

    def test_wave7_worker_03_04_05_07_network_04_05_06_sec_04_05_07_14_and_comm_12(self):
        import tempfile
        from aix_zero.network_coordinator import (
            ImmutableArtifactStore,
            WorkerEnrollmentRegistry,
            WorkerHeartbeatAndCancellationTracker,
            build_verified_contribution_ledger,
            enforce_worker_resource_limits,
            evaluate_independent_result_replication,
            generate_supply_chain_sbom_review,
            manage_worker_disk_quota,
            manage_worker_schedule_and_pause,
            match_job_to_worker_capabilities,
            validate_constrained_job_arguments,
            verify_dataset_distribution_permission,
        )

        # WORKER-03 CPU and memory limits
        r_ok = enforce_worker_resource_limits(
            max_cpu_threads=1, max_memory_mb=512, job_requested_threads=1, job_estimated_memory_mb=256
        )
        self.assertTrue(r_ok["execution_allowed"])
        r_stress = enforce_worker_resource_limits(
            max_cpu_threads=1, max_memory_mb=512, job_requested_threads=4, job_estimated_memory_mb=1024
        )
        self.assertFalse(r_stress["execution_allowed"])
        self.assertEqual(r_stress["status"], "TERMINATED_RESOURCE_CAP_EXCEEDED")

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)

            # WORKER-04 contribution schedules and pause controls survive restarts
            sched_cfg = tdir / "worker_schedule.json"
            s1 = manage_worker_schedule_and_pause(
                sched_cfg,
                allowed_hours_utc=[10, 11, 12],
                idle_only=True,
                paused=True,
                active_lease_policy="RELINQUISH_CLEANLY",
                current_hour_utc=11,
                host_is_idle=True,
                active_lease_id="lease-active-99",
            )
            self.assertFalse(s1["can_accept_new_lease"])
            self.assertEqual(s1["active_lease_action"], "RELINQUISH_CLEANLY")
            # Restart reload without args preserves paused=True and resource caps
            s2 = manage_worker_schedule_and_pause(sched_cfg, current_hour_utc=11, host_is_idle=True)
            self.assertFalse(s2["can_accept_new_lease"])
            self.assertEqual(s2["persisted_settings"]["max_cpu_threads"], 2)

            # WORKER-05 bounded disk quota and cleanup
            wdir = tdir / "worker_storage"
            wdir.mkdir()
            outside_file = tdir / "unrelated_user_file.txt"
            outside_file.write_text("do-not-touch", encoding="utf-8")
            (wdir / "job1.scratch").write_bytes(b"x" * 100)
            (wdir / "pending_result.json").write_bytes(b"y" * 80)

            dq = manage_worker_disk_quota(
                wdir,
                quota_bytes=150,
                incoming_artifact_bytes=50,
                pending_upload_filenames=["pending_result.json"],
            )
            self.assertTrue(dq["work_permitted"])
            self.assertIn("job1.scratch", dq["reclaimed_files"])
            self.assertIn("pending_result.json", dq["retained_pending_files"])
            self.assertTrue(outside_file.exists())

            dq_exceed = manage_worker_disk_quota(
                wdir,
                quota_bytes=100,
                incoming_artifact_bytes=50,
                pending_upload_filenames=["pending_result.json"],
            )
            self.assertFalse(dq_exceed["work_permitted"])
            self.assertEqual(dq_exceed["status"], "STOPPED_DISK_QUOTA_EXCEEDED")
            self.assertTrue(outside_file.exists())

            # WORKER-07 heartbeat and idempotent lease cancellation
            hb = WorkerHeartbeatAndCancellationTracker(tdir / "hb.json")
            hb.register_lease("job-1", "lease-1", expires_at_epoch=100)
            self.assertTrue(hb.record_heartbeat("lease-1", now_epoch=50)["lease_active"])
            c1 = hb.cancel_lease("lease-1", reason="user_paused")
            self.assertFalse(c1["idempotent_noop"])
            c2 = hb.cancel_lease("lease-1", reason="duplicate_cancel")
            self.assertTrue(c2["idempotent_noop"])
            self.assertEqual(c2["cancellation_log_count"], 1)
            with self.assertRaisesRegex(PermissionError, "Lost or cancelled lease"):
                hb.validate_upload_eligibility("lease-1", now_epoch=60)

            # NETWORK-05 immutable input and result artifact store
            store = ImmutableArtifactStore(tdir / "immutable_store")
            with self.assertRaises(RuntimeError):
                store.publish_result_artifact(
                    b"incomplete-payload",
                    model_sha256="a" * 64,
                    dataset_sha256="b" * 64,
                    source_sha256="c" * 64,
                    config_sha256="d" * 64,
                    simulate_incomplete_staging=True,
                )
            self.assertEqual(list(store.objects_dir.glob("*.bin")), [])
            man = store.publish_result_artifact(
                b"verified-result-payload",
                model_sha256="a" * 64,
                dataset_sha256="b" * 64,
                source_sha256="c" * 64,
                config_sha256="d" * 64,
            )
            self.assertEqual(store.retrieve_artifact(man["object_sha256"]), b"verified-result-payload")

        # SEC-05 constrained public job arguments and paths
        with self.assertRaisesRegex(ValueError, "Shell metacharacters"):
            validate_constrained_job_arguments(
                {"task": "BP", "artifact_relative_path": "out.json", "arguments": {"arg": "1; rm -rf /"}},
                user_resource_caps={"max_memory_mb": 512},
            )
        with self.assertRaisesRegex(ValueError, "cannot alter user-selected resource caps"):
            validate_constrained_job_arguments(
                {"task": "BP", "artifact_relative_path": "out.json", "max_memory_mb": 999999},
                user_resource_caps={"max_memory_mb": 512},
            )
        valid_args = validate_constrained_job_arguments(
            {"task": "BP", "artifact_relative_path": "results/out.json", "arguments": {"seed": "42"}},
            user_resource_caps={"max_memory_mb": 512},
        )
        self.assertTrue(valid_args["validated"])

        # NETWORK-06 independent result replication
        rep_disagree = evaluate_independent_result_replication(
            {"worker_id": "w1", "accuracy": 0.9, "result_sha256": "h1"},
            {"worker_id": "w2", "accuracy": 0.1, "result_sha256": "h2"},
        )
        self.assertEqual(rep_disagree["replication_status"], "DISAGREEMENT_INVESTIGATION_REQUIRED")
        self.assertEqual(rep_disagree["useful_verified_throughput_ratio"], 0.0)

        rep_agree = evaluate_independent_result_replication(
            {"worker_id": "w1", "accuracy": 0.125, "result_sha256": "h1"},
            {"worker_id": "w2", "accuracy": 0.125, "result_sha256": "h1"},
        )
        self.assertEqual(rep_agree["replication_status"], "VERIFIED_BY_REPLICATION")
        self.assertEqual(rep_agree["useful_verified_throughput_ratio"], 0.5)

        # SEC-04 authenticated worker enrollment and revocation
        reg = WorkerEnrollmentRegistry(master_secret="coord-secret")
        w_cred = reg.enroll_worker("worker-01", expires_at_epoch=500)
        self.assertTrue(reg.verify_worker_can_lease("worker-01", w_cred["enrollment_token"], now_epoch=100))
        reg.revoke_worker("worker-01", reason="compromised_key")
        with self.assertRaisesRegex(PermissionError, "Revoked worker"):
            reg.verify_worker_can_lease("worker-01", w_cred["enrollment_token"], now_epoch=110)
        pub_tel = reg.export_public_worker_telemetry()
        self.assertNotIn("enrollment_token", pub_tel[0])

        # NETWORK-04 match work to hardware and contributor budgets
        m_defer = match_job_to_worker_capabilities(
            {"min_ram_mb": 2048, "min_disk_mb": 500, "allowed_backends": ["cuda"], "estimated_duration_seconds": 300},
            {"available_ram_mb": 1024, "available_disk_mb": 1000, "accelerator_backend": "cpu", "max_runtime_seconds": 120},
        )
        self.assertFalse(m_defer["assigned"])
        self.assertIn("insufficient_ram_mb", m_defer["deferral_reasons"])
        self.assertIn("incompatible_accelerator_backend", m_defer["deferral_reasons"])

        # SEC-07 training-data distribution permissions
        with self.assertRaisesRegex(PermissionError, "private sample data"):
            verify_dataset_distribution_permission(
                {"license": "MIT", "provenance_sha256": "a" * 64, "public_redistribution_allowed": True, "contains_private_samples": True}
            )
        self.assertTrue(
            verify_dataset_distribution_permission(
                {"license": "MIT", "provenance_sha256": "a" * 64, "public_redistribution_allowed": True, "contains_private_samples": False}
            )["distribution_permitted"]
        )

        # SEC-14 lightweight dependency and supply-chain review
        with self.assertRaisesRegex(ValueError, "Unreviewed dependency update rejected"):
            generate_supply_chain_sbom_review(
                dependencies=[{"name": "torch", "pinned_version": "2.6.0", "provenance": "pypi-sha256"}],
                known_unresolved_issues=[],
                unreviewed_dependency_update=True,
            )
        sbom = generate_supply_chain_sbom_review(
            dependencies=[
                {"name": "torch", "pinned_version": "2.6.0", "provenance": "pypi-wheel-sha256"},
                {"name": "safetensors", "pinned_version": "0.5.3", "provenance": "pypi-wheel-sha256"},
            ],
            known_unresolved_issues=["None blocking CPU inference"],
        )
        self.assertTrue(sbom["signing_gate_passed"])

        # COMM-12 recognize verified contributions fairly & opt-out attribution
        ledger = build_verified_contribution_ledger(
            [
                {
                    "contribution_id": "c-1",
                    "contribution_type": "independent_reproduction",
                    "verification_status": "ACCEPTED_VERIFIED",
                    "contributor_handle": "alice",
                    "opt_out_public_attribution": False,
                    "accepted_useful_work_units": 1,
                    "self_reported_runtime_hours": 9999,
                    "artifact_sha256": "a" * 64,
                },
                {
                    "contribution_id": "c-2",
                    "contribution_type": "verified_compute",
                    "verification_status": "ACCEPTED_VERIFIED",
                    "contributor_handle": "bob",
                    "opt_out_public_attribution": True,
                    "accepted_useful_work_units": 2,
                    "artifact_sha256": "b" * 64,
                },
                {
                    "contribution_id": "c-3",
                    "contribution_type": "verified_compute",
                    "verification_status": "UNVERIFIED_SELF_REPORT",
                    "contributor_handle": "mallory",
                    "self_reported_runtime_hours": 10000,
                },
            ]
        )
        self.assertEqual(ledger["recognized_contribution_count"], 2)
        self.assertEqual(ledger["rejected_unverified_count"], 1)
        self.assertEqual(ledger["recognized_contributions"][1]["public_contributor_handle"], "anonymous-contributor")

    def test_wave8_worker_06_08_10_network_07_09_and_comm_09(self):
        import hashlib
        import tempfile
        from aix_zero.network_coordinator import (
            resume_bounded_worker_job,
            schedule_with_fairness_and_starvation_control,
            validate_and_ingest_worker_result,
            validate_opt_in_research_proposal,
            verify_desktop_client_package_lifecycle,
            verify_worker_input_artifact_before_execution,
        )

        # WORKER-06 verify input artifacts before execution
        full_bytes = b"verified-input-dataset-payload"
        full_sha = hashlib.sha256(full_bytes).hexdigest()
        trunc = verify_worker_input_artifact_before_execution(
            [full_bytes[:10]], expected_sha256=full_sha, expected_byte_length=len(full_bytes)
        )
        self.assertFalse(trunc["execution_permitted"])
        self.assertEqual(trunc["status"], "TRUNCATED_NEEDS_RESUMABLE_FETCH")
        self.assertEqual(trunc["resume_byte_offset"], 10)

        corrupt = verify_worker_input_artifact_before_execution(
            [b"x" * len(full_bytes)], expected_sha256=full_sha, expected_byte_length=len(full_bytes)
        )
        self.assertFalse(corrupt["execution_permitted"])
        self.assertEqual(corrupt["status"], "REJECTED_CORRUPT_SHA256_MISMATCH")

        ready = verify_worker_input_artifact_before_execution(
            [full_bytes[:10], full_bytes[10:]], expected_sha256=full_sha, expected_byte_length=len(full_bytes)
        )
        self.assertTrue(ready["execution_permitted"])

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            src_ckpt = tdir / "source_v11.safetensors"
            out_ckpt = tdir / "candidate_out.safetensors"

            # WORKER-08 resumable bounded jobs never overwrite source models
            with self.assertRaisesRegex(PermissionError, "prohibited from overwriting immutable source model"):
                resume_bounded_worker_job(
                    checkpoint_bundle={"experiment_config_sha256": "a" * 64, "step": 10},
                    active_job_contract={"experiment_config_sha256": "a" * 64},
                    source_model_path=src_ckpt,
                    candidate_output_path=src_ckpt,
                )
            res_ok = resume_bounded_worker_job(
                checkpoint_bundle={"experiment_config_sha256": "a" * 64, "step": 10},
                active_job_contract={"experiment_config_sha256": "a" * 64},
                source_model_path=src_ckpt,
                candidate_output_path=out_ckpt,
                lease_active=True,
            )
            self.assertTrue(res_ok["resumed"])
            self.assertEqual(res_ok["resumed_from_step"], 10)

            # WORKER-10 desktop client package install/uninstall and GPU backend guard
            gpu_fail = verify_desktop_client_package_lifecycle(
                tdir / "pkg_gpu", requested_gpu_backend="rocm", gpu_backend_supported=False
            )
            self.assertFalse(gpu_fail["installed"])
            self.assertFalse(gpu_fail["silent_excessive_cpu_fallback"])

            pkg_ok = verify_desktop_client_package_lifecycle(tdir / "pkg_cpu", requested_gpu_backend="cpu")
            self.assertTrue(pkg_ok["installed"])
            self.assertTrue(pkg_ok["clean_uninstall_verified"])

        # NETWORK-07 reject malformed and self-declared worker results without crashing coordinator
        rej_log = []
        self_dec = validate_and_ingest_worker_result(
            {"job_id": "j1", "promotion_status": "PROMOTED"}, rejection_log=rej_log
        )
        self.assertFalse(self_dec["accepted"])
        self.assertTrue(self_dec["coordinator_healthy"])
        self.assertEqual(len(rej_log), 1)

        ing_ok = validate_and_ingest_worker_result(
            {
                "job_id": "j1",
                "lease_id": "l1",
                "worker_id": "w1",
                "result_sha256": "a" * 64,
                "metrics": {"accuracy": 0.125},
            },
            rejection_log=rej_log,
        )
        self.assertTrue(ing_ok["accepted"])

        # NETWORK-09 fairness and starvation controls
        sched = schedule_with_fairness_and_starvation_control(
            [
                {"worker_id": "fast-monopoly", "active_leases": 2, "wait_seconds": 10, "reliability_score": 1.0},
                {"worker_id": "fast-idle", "active_leases": 0, "wait_seconds": 20, "reliability_score": 1.0},
                {"worker_id": "slow-reliable-starved", "active_leases": 0, "wait_seconds": 400, "reliability_score": 0.95},
            ],
            max_active_leases_per_contributor=2,
            starvation_threshold_seconds=300,
        )
        self.assertEqual(sched["selected_worker_id"], "slow-reliable-starved")
        self.assertIn("fast-monopoly", sched["throttled_workers"])
        self.assertTrue(sched["starvation_boost_applied"])

        # COMM-09 opt-in community research discussion process
        with self.assertRaisesRegex(ValueError, "unsolicited mass mentions"):
            validate_opt_in_research_proposal(
                {
                    "hypothesis": "H1",
                    "controls": ["base"],
                    "validation_budget": "100 steps",
                    "stopping_criteria": "2 failures",
                    "uses_unsolicited_mass_mentions": True,
                }
            )
        prop_ok = validate_opt_in_research_proposal(
            {
                "hypothesis": "H1",
                "controls": ["base", "prior_only"],
                "validation_budget": "180 steps",
                "stopping_criteria": "paired Hoeffding gate",
            }
        )
        self.assertTrue(prop_ok["proposal_valid"])
        self.assertTrue(prop_ok["opt_in_only"])


if __name__ == "__main__":
    unittest.main()
