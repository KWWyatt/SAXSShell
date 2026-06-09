import json

from saxshell.saxs.project_manager import (
    PROJECT_LAYOUT_VERSION_CURRENT,
    PROJECT_LAYOUT_VERSION_LEGACY,
    SAXSProjectManager,
    build_project_paths,
    project_artifact_paths,
)


def test_project_paths_support_clean_layout_version(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "clean_project"

    settings = manager.create_project(
        project_dir,
        project_layout_version=PROJECT_LAYOUT_VERSION_CURRENT,
    )
    paths = build_project_paths(project_dir)
    artifacts = project_artifact_paths(settings)

    assert settings.project_layout_version == PROJECT_LAYOUT_VERSION_CURRENT
    assert paths.project_layout_version == PROJECT_LAYOUT_VERSION_CURRENT
    assert (
        paths.experimental_data_dir
        == (project_dir / "inputs" / "experimental_data").resolve()
    )
    assert (
        paths.saved_distributions_dir
        == (project_dir / "analysis" / "computed_distributions").resolve()
    )
    assert (
        paths.exported_data_dir == (project_dir / "outputs" / "data").resolve()
    )
    assert artifacts.component_dir == paths.scattering_components_dir
    assert artifacts.plots_dir == paths.plots_dir
    assert artifacts.prefit_dir == paths.prefit_dir
    assert artifacts.dream_dir == paths.dream_dir
    assert (
        json.loads(paths.project_file.read_text(encoding="utf-8"))[
            "project_layout_version"
        ]
        == PROJECT_LAYOUT_VERSION_CURRENT
    )


def test_missing_layout_version_loads_as_legacy_project(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "legacy_project"
    settings = manager.create_project(project_dir)
    paths = build_project_paths(project_dir)
    payload = settings.to_dict()
    payload.pop("project_layout_version")
    paths.project_file.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = manager.load_project(project_dir)
    loaded_paths = build_project_paths(project_dir)

    assert loaded.project_layout_version == PROJECT_LAYOUT_VERSION_LEGACY
    assert (
        loaded_paths.saved_distributions_dir
        == (project_dir / "saved_distributions").resolve()
    )
    assert (
        loaded_paths.experimental_data_dir
        == (project_dir / "experimental_data").resolve()
    )


def test_clean_layout_can_read_legacy_saved_distribution_root(tmp_path):
    manager = SAXSProjectManager()
    project_dir = tmp_path / "mixed_project"
    settings = manager.create_project(
        project_dir,
        project_layout_version=PROJECT_LAYOUT_VERSION_CURRENT,
    )
    legacy_distribution_dir = (
        build_project_paths(
            project_dir,
            project_layout_version=PROJECT_LAYOUT_VERSION_LEGACY,
        ).saved_distributions_dir
        / "dist_legacy"
    )
    legacy_distribution_dir.mkdir(parents=True)
    (legacy_distribution_dir / "distribution.json").write_text(
        json.dumps(
            {
                "distribution_id": "dist_legacy",
                "label": "Legacy distribution",
                "component_artifacts_ready": False,
                "prior_artifacts_ready": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    records = manager.list_saved_distributions(project_dir)
    settings.active_distribution_id = "dist_legacy"
    artifact_paths = project_artifact_paths(settings)

    assert [record.distribution_id for record in records] == ["dist_legacy"]
    assert records[0].distribution_dir == legacy_distribution_dir.resolve()
    assert artifact_paths.root_dir == legacy_distribution_dir.resolve()
