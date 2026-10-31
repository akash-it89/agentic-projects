import pytest

from p3docintel import config


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Every test writes into its own temp folder and ignores the developer's .env."""
    settings = config.Settings(runs_dir=tmp_path / "runs", _env_file=None)
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    for mod in ("store", "pipeline", "api", "chat", "ocr"):
        module = __import__(f"p3docintel.{mod}", fromlist=["get_settings"])
        if hasattr(module, "get_settings"):
            monkeypatch.setattr(module, "get_settings", lambda: settings)
    return settings


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    from p3docintel.samples import generate

    out = tmp_path_factory.mktemp("samples")
    generate(out)
    return out
