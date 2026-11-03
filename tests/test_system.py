import socket

from tracking.system import system_fingerprint


def test_system_fingerprint_has_expected_keys():
    info = system_fingerprint()
    for key in (
        "hostname", "os", "kernel", "arch", "cpu_count",
        "python_version", "python_implementation", "executable", "cwd",
    ):
        assert key in info


def test_opt_out_returns_empty_dict(monkeypatch):
    monkeypatch.setenv("TRACKING_SYSTEM_INFO", "0")
    assert system_fingerprint() == {}


def test_slurm_fields_only_present_when_job_id_set(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert "slurm" not in system_fingerprint()

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_NODELIST", "node[01-02]")
    info = system_fingerprint()
    assert info["slurm"] == {"job_id": "12345", "nodelist": "node[01-02]"}


def test_a_failing_probe_does_not_raise_and_others_still_populate(monkeypatch):
    def boom():
        raise RuntimeError("no hostname for you")

    monkeypatch.setattr(socket, "gethostname", boom)
    info = system_fingerprint()
    assert "hostname" not in info
    assert "os" in info
    assert "cwd" in info
