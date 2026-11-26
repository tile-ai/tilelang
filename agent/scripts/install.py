#!/usr/bin/env python3
"""Render and install TileLang tuning agents for supported coding tools."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11
    tomllib = None


SUPPORTED_TOOLS = ("codex", "opencode", "claude")
INSTALL_MANIFEST = "tilelang-tuning-manifest.json"
PLATFORM_TOKENS = (".codex/", ".opencode/", ".claude/", "Codex subagent")

SCRIPT_PATH = Path(__file__).resolve()
AGENT_ROOT = SCRIPT_PATH.parents[1]
SOURCE_REPO_ROOT = AGENT_ROOT.parent
SOURCE_MANIFEST_PATH = AGENT_ROOT / "manifest.json"
SOURCE_SKILLS_ROOT = SOURCE_REPO_ROOT / ".agents" / "skills" / "tilelang-ascend-op"


class InstallError(RuntimeError):
    pass


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: str
    content: bytes | None = None
    link_target: str | None = None

    @property
    def fingerprint(self) -> str:
        if self.kind == "file":
            assert self.content is not None
            return "file:" + hashlib.sha256(self.content).hexdigest()
        if self.kind == "symlink":
            assert self.link_target is not None
            return "symlink:" + self.link_target
        raise InstallError(f"unsupported artifact kind: {self.kind}")


@dataclass(frozen=True)
class TargetLayout:
    tool: str
    scope: str
    install_base: Path
    config_root: Path
    agent_root: Path
    workflow_root: Path
    skill_root: Path

    @property
    def manifest_path(self) -> Path:
        return self.config_root / INSTALL_MANIFEST


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstallError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise InstallError(f"JSON root must be an object: {path}")
    return data


def source_manifest() -> dict[str, Any]:
    return read_json(SOURCE_MANIFEST_PATH)


def target_layout(tool: str, scope: str, requested_path: str | None) -> TargetLayout:
    home = Path.home()
    if scope == "project":
        install_base = Path(requested_path).expanduser().resolve() if requested_path else AGENT_ROOT
        if not install_base.is_dir():
            raise InstallError(f"project path is not a directory: {install_base}")
        config_root = install_base / f".{tool}"
    else:
        if requested_path:
            raise InstallError("--path can only be used with --scope project")
        install_base = home
        if tool == "codex":
            config_root = Path(os.environ.get("CODEX_HOME", home / ".codex")).expanduser()
        elif tool == "opencode":
            xdg_config = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config")).expanduser()
            config_root = xdg_config / "opencode"
        else:
            config_root = home / ".claude"

    if tool in ("codex", "opencode"):
        shared_skill_root = install_base / ".agents" / "skills" if scope == "project" else home / ".agents" / "skills"
        skill_root = shared_skill_root / "tilelang-ascend-op"
    else:
        skill_root = config_root / "skills"

    return TargetLayout(
        tool=tool,
        scope=scope,
        install_base=install_base,
        config_root=config_root,
        agent_root=config_root / "agents",
        workflow_root=config_root / "workflows",
        skill_root=skill_root,
    )


def yaml_scalar(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def load_template(relative_path: str) -> str:
    path = AGENT_ROOT / "adapters" / relative_path
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InstallError(f"cannot read adapter template {path}: {exc}") from exc


def render_codex(agent: dict[str, Any], body: str) -> tuple[bytes, bytes]:
    template = load_template("codex/agent.toml.tpl")
    name = str(agent["name"])
    rendered = (
        template.replace("__NAME__", json.dumps(name, ensure_ascii=False))
        .replace("__DESCRIPTION__", json.dumps(str(agent["description"]), ensure_ascii=False))
        .replace("__AGENT_FILE__", name)
    )
    return rendered.encode("utf-8"), body.encode("utf-8")


def render_opencode(agent: dict[str, Any], body: str, schema: str) -> bytes:
    role = str(agent["role"])
    template = load_template(f"opencode/{schema}-{role}.md.tpl")
    children = [str(item) for item in agent.get("children", [])]
    child_permissions = ""
    if children:
        if schema == "classic":
            child_permissions = "\n".join(f"    {yaml_scalar(child)}: allow" for child in children)
        else:
            child_permissions = "\n".join(
                "\n".join(
                    (
                        "  - action: subagent",
                        f"    resource: {yaml_scalar(child)}",
                        "    effect: allow",
                    )
                )
                for child in children
            )
    rendered = (
        template.replace("__DESCRIPTION__", yaml_scalar(str(agent["description"])))
        .replace("__CHILD_PERMISSIONS__", child_permissions)
        .replace("__BODY__", body.rstrip() + "\n")
    )
    return rendered.encode("utf-8")


def render_claude(agent: dict[str, Any], body: str) -> bytes:
    role = str(agent["role"])
    template = load_template(f"claude/{role}.md.tpl")
    children = ", ".join(str(item) for item in agent.get("children", []))
    rendered = (
        template.replace("__NAME__", str(agent["name"]))
        .replace("__DESCRIPTION__", yaml_scalar(str(agent["description"])))
        .replace("__CHILDREN__", children)
        .replace("__BODY__", body.rstrip() + "\n")
    )
    return rendered.encode("utf-8")


def build_artifacts(
    layout: TargetLayout,
    manifest: dict[str, Any],
    opencode_schema: str = "classic",
) -> list[Artifact]:
    artifacts: list[Artifact] = []

    # Project installs use the selected directory as an isolated tuning entry.
    # AGENTS.md activates Codex and OpenCode, while Claude imports the same
    # bootstrap through its native CLAUDE.md project-instructions file.
    if layout.scope == "project":
        artifacts.append(
            Artifact(
                layout.install_base / "AGENTS.md",
                "file",
                content=load_template("common/AGENTS.md.tpl").encode("utf-8"),
            )
        )
        if layout.tool == "claude":
            artifacts.append(
                Artifact(
                    layout.install_base / "CLAUDE.md",
                    "file",
                    content=load_template("claude/CLAUDE.md.tpl").encode("utf-8"),
                )
            )

    for agent in manifest["agents"]:
        source = AGENT_ROOT / str(agent["source"])
        body = source.read_text(encoding="utf-8")
        name = str(agent["name"])
        if layout.tool == "codex":
            toml_content, body_content = render_codex(agent, body)
            artifacts.append(Artifact(layout.agent_root / f"{name}.toml", "file", content=toml_content))
            artifacts.append(Artifact(layout.agent_root / f"{name}.md", "file", content=body_content))
        elif layout.tool == "opencode":
            artifacts.append(
                Artifact(
                    layout.agent_root / f"{name}.md",
                    "file",
                    content=render_opencode(agent, body, opencode_schema),
                )
            )
        else:
            artifacts.append(Artifact(layout.agent_root / f"{name}.md", "file", content=render_claude(agent, body)))

    for workflow_source in manifest["workflows"]:
        source = AGENT_ROOT / str(workflow_source)
        artifacts.append(Artifact(layout.workflow_root / source.name, "file", content=source.read_bytes()))

    source_skill_root = SOURCE_SKILLS_ROOT.resolve()
    target_skill_root = layout.skill_root.resolve(strict=False)
    if source_skill_root != target_skill_root:
        for skill in manifest["skills"]:
            source = (SOURCE_SKILLS_ROOT / str(skill)).resolve()
            artifacts.append(
                Artifact(
                    layout.skill_root / str(skill),
                    "symlink",
                    link_target=str(source),
                )
            )

    return artifacts


def current_fingerprint(path: Path) -> str | None:
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    if path.is_file():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return "file:" + digest
    if path.exists():
        return "directory"
    return None


def load_install_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def owned_fingerprints(install_manifest: dict[str, Any] | None) -> dict[str, str]:
    if not install_manifest:
        return {}
    result: dict[str, str] = {}
    for entry in install_manifest.get("artifacts", []):
        if isinstance(entry, dict) and isinstance(entry.get("path"), str) and isinstance(entry.get("fingerprint"), str):
            result[entry["path"]] = entry["fingerprint"]
    return result


def validate_install_manifest_target(
    install_manifest: dict[str, Any],
    layout: TargetLayout,
    source: dict[str, Any],
) -> None:
    expected = {
        "package": source["package"],
        "tool": layout.tool,
        "scope": layout.scope,
    }
    mismatches = [
        f"{key}={install_manifest.get(key)!r} (expected {value!r})" for key, value in expected.items() if install_manifest.get(key) != value
    ]
    if mismatches:
        raise InstallError(f"install manifest does not belong to this target: {layout.manifest_path}\n  " + "\n  ".join(mismatches))


def backup_path(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}.bak.{timestamp}")
    suffix = 1
    while candidate.exists() or candidate.is_symlink():
        candidate = path.with_name(f"{path.name}.bak.{timestamp}.{suffix}")
        suffix += 1
    return candidate


def remove_replaceable(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.exists():
        raise InstallError(f"refusing to remove a real directory with --force: {path}")


def write_artifact(artifact: Artifact) -> None:
    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    if artifact.path.exists() or artifact.path.is_symlink():
        remove_replaceable(artifact.path)

    if artifact.kind == "file":
        assert artifact.content is not None
        handle, temporary_name = tempfile.mkstemp(prefix=f".{artifact.path.name}.", dir=artifact.path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(artifact.content)
            temporary.chmod(0o644)
            os.replace(temporary, artifact.path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return

    assert artifact.link_target is not None
    temporary = artifact.path.with_name(f".{artifact.path.name}.link.{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.symlink(artifact.link_target, temporary)
    os.replace(temporary, artifact.path)


def make_install_manifest(
    layout: TargetLayout,
    source: dict[str, Any],
    artifacts: list[Artifact],
    opencode_schema: str,
) -> dict[str, Any]:
    result = {
        "package": source["package"],
        "version": source["version"],
        "tool": layout.tool,
        "scope": layout.scope,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(AGENT_ROOT),
        "artifacts": [
            {
                "path": str(artifact.path),
                "kind": artifact.kind,
                "fingerprint": artifact.fingerprint,
            }
            for artifact in artifacts
        ],
    }
    if layout.tool == "opencode":
        result["adapter_options"] = {"schema": opencode_schema}
    return result


def detect_opencode_schema() -> str:
    """Select the installed OpenCode configuration generation when possible."""
    binary = shutil.which("opencode")
    if binary is None:
        return "classic"
    try:
        completed = subprocess.run(
            [binary, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "classic"
    match = re.search(r"\d+", completed.stdout + " " + completed.stderr)
    if match and int(match.group()) >= 2:
        return "v2"
    return "classic"


def opencode_schema_for(
    layout: TargetLayout,
    requested: str,
    *,
    preserve_installed: bool = False,
) -> str:
    if layout.tool != "opencode":
        return "classic"
    if requested != "auto":
        return requested
    if preserve_installed:
        installed = load_install_manifest(layout.manifest_path)
        if installed is not None:
            schema = installed.get("adapter_options", {}).get("schema")
            if schema in ("classic", "v2"):
                return str(schema)
    return detect_opencode_schema()


def claimed_by_another_tool(layout: TargetLayout, path: Path) -> bool:
    requested_path = str(layout.install_base) if layout.scope == "project" else None
    for tool in SUPPORTED_TOOLS:
        if tool == layout.tool:
            continue
        other_layout = target_layout(tool, layout.scope, requested_path)
        other_manifest = load_install_manifest(other_layout.manifest_path)
        if other_manifest is None:
            continue
        for entry in other_manifest.get("artifacts", []):
            if isinstance(entry, dict) and entry.get("path") == str(path):
                return True
    return False


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    content = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    write_artifact(Artifact(path, "file", content=content))


def install(
    layout: TargetLayout,
    source: dict[str, Any],
    args: argparse.Namespace,
    opencode_schema: str,
) -> None:
    artifacts = build_artifacts(layout, source, opencode_schema)
    previous = load_install_manifest(layout.manifest_path)
    if previous is not None:
        validate_install_manifest_target(previous, layout, source)
    owned = owned_fingerprints(previous)
    plan: list[tuple[str, Artifact]] = []
    conflicts: list[Path] = []

    for artifact in artifacts:
        current = current_fingerprint(artifact.path)
        if current == artifact.fingerprint:
            plan.append(("unchanged", artifact))
        elif current is None:
            plan.append(("create", artifact))
        elif owned.get(str(artifact.path)) == current:
            plan.append(("update", artifact))
        elif args.backup_existing:
            plan.append(("backup+replace", artifact))
        elif args.force and current != "directory":
            plan.append(("force-replace", artifact))
        else:
            conflicts.append(artifact.path)

    # Retire directory links for skills removed from the install manifest.
    # Only unchanged, installer-owned links may be removed; source skills remain intact.
    artifact_paths = {str(artifact.path) for artifact in artifacts}
    retired_skills: list[tuple[str, Path]] = []
    for entry in (previous or {}).get("artifacts", []):
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        path = Path(entry["path"])
        if entry.get("kind") != "symlink" or path.parent != layout.skill_root or str(path) in artifact_paths:
            continue
        current = current_fingerprint(path)
        if current is None:
            continue
        if not path.is_symlink() or current != owned.get(str(path)):
            action = "retain changed"
        elif claimed_by_another_tool(layout, path):
            action = "retain shared"
        else:
            action = "remove retired"
        retired_skills.append((action, path))

    adapter_note = f"; schema={opencode_schema}" if layout.tool == "opencode" else ""
    print(f"[{layout.tool}] {layout.scope} install at {layout.config_root}{adapter_note}")
    for action, artifact in plan:
        print(f"  {action:14} {artifact.path}")
    for action, path in retired_skills:
        print(f"  {action:14} {path}")

    if conflicts:
        joined = "\n".join(f"  - {path}" for path in conflicts)
        raise InstallError(
            "unmanaged or modified targets would be replaced:\n"
            f"{joined}\nUse --backup-existing to preserve them or --force for regular files/symlinks."
        )
    if args.dry_run:
        return

    for action, artifact in plan:
        if action == "unchanged":
            continue
        if action == "backup+replace":
            backup = backup_path(artifact.path)
            artifact.path.rename(backup)
            print(f"  backed up      {artifact.path} -> {backup}")
        write_artifact(artifact)

    for action, path in retired_skills:
        if action == "remove retired":
            path.unlink()
    write_json_atomic(
        layout.manifest_path,
        make_install_manifest(layout, source, artifacts, opencode_schema),
    )
    print(f"  manifest       {layout.manifest_path}")


def uninstall(layout: TargetLayout, source: dict[str, Any], dry_run: bool) -> None:
    manifest = load_install_manifest(layout.manifest_path)
    if manifest is None:
        print(f"[{layout.tool}] no install manifest: {layout.manifest_path}")
        return
    validate_install_manifest_target(manifest, layout, source)

    expected_paths = {str(artifact.path) for artifact in build_artifacts(layout, source)}
    invalid_entries = [
        entry
        for entry in manifest.get("artifacts", [])
        if (
            not isinstance(entry, dict)
            or entry.get("path") not in expected_paths
            or entry.get("kind") not in ("file", "symlink")
            or not isinstance(entry.get("fingerprint"), str)
        )
    ]
    if invalid_entries:
        details = "\n".join(f"  - {entry!r}" for entry in invalid_entries)
        raise InstallError("refusing to uninstall invalid entries or paths not declared by the current source manifest:\n" + details)

    retained: list[dict[str, Any]] = []
    for entry in manifest.get("artifacts", []):
        path = Path(str(entry["path"]))
        expected = str(entry["fingerprint"])
        current = current_fingerprint(path)
        if current is None:
            print(f"  already absent {path}")
        elif current == expected:
            if claimed_by_another_tool(layout, path):
                print(f"  retain shared  {path}")
            else:
                print(f"  remove         {path}")
                if not dry_run:
                    remove_replaceable(path)
        else:
            print(f"  retain changed {path}")
            retained.append(entry)

    if dry_run:
        return
    if retained:
        manifest["artifacts"] = retained
        manifest["status"] = "partial-uninstall"
        write_json_atomic(layout.manifest_path, manifest)
        print(f"  partial manifest retained at {layout.manifest_path}")
    else:
        layout.manifest_path.unlink(missing_ok=True)
        for directory in (layout.agent_root, layout.workflow_root, layout.skill_root):
            with contextlib.suppress(OSError):
                directory.rmdir()
        print(f"  removed manifest {layout.manifest_path}")


def validate_markdown_links(path: Path, errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        reference = match.group(1).split("#", 1)[0]
        if not reference or reference.startswith(("http://", "https://", "/")):
            continue
        resolved = (path.parent / reference).resolve()
        if not resolved.exists():
            errors.append(f"broken Markdown reference in {path}: {reference}")


def validate_source(manifest: dict[str, Any]) -> None:
    errors: list[str] = []
    if manifest.get("package") != "tilelang-tuning":
        errors.append("manifest package must be tilelang-tuning")
    agents = manifest.get("agents")
    workflows = manifest.get("workflows")
    skills = manifest.get("skills")
    if not isinstance(agents, list) or not agents:
        errors.append("manifest agents must be a non-empty list")
        agents = []
    if not isinstance(workflows, list) or not workflows:
        errors.append("manifest workflows must be a non-empty list")
        workflows = []
    if not isinstance(skills, list) or not skills:
        errors.append("manifest skills must be a non-empty list")
        skills = []

    names = [item.get("name") for item in agents if isinstance(item, dict)]
    if len(names) != len(set(names)):
        errors.append("agent names must be unique")
    known_names = set(names)

    source_files: list[Path] = []
    for agent in agents:
        if not isinstance(agent, dict):
            errors.append("each agent entry must be an object")
            continue
        if agent.get("role") not in ("primary", "subagent"):
            errors.append(f"invalid role for {agent.get('name')}")
        source = AGENT_ROOT / str(agent.get("source", ""))
        if not source.is_file():
            errors.append(f"missing agent source: {source}")
        else:
            source_files.append(source)
        for child in agent.get("children", []):
            if child not in known_names:
                errors.append(f"unknown child agent {child} in {agent.get('name')}")

    for workflow in workflows:
        source = AGENT_ROOT / str(workflow)
        if not source.is_file():
            errors.append(f"missing workflow source: {source}")
        else:
            source_files.append(source)

    for skill in skills:
        skill_file = SOURCE_SKILLS_ROOT / str(skill) / "SKILL.md"
        if not skill_file.is_file():
            errors.append(f"missing skill: {skill_file}")

    referenced_skills: set[str] = set()
    for source in source_files:
        text = source.read_text(encoding="utf-8")
        for token in PLATFORM_TOKENS:
            if token in text:
                errors.append(f"platform-specific token {token!r} in {source}")
        referenced_skills.update(re.findall(r"\$([a-z][a-z0-9-]*)", text))
        validate_markdown_links(source, errors)
    missing_skills = referenced_skills - set(str(skill) for skill in skills)
    if missing_skills:
        errors.append(f"skills referenced by core but absent from manifest: {sorted(missing_skills)}")

    required_templates = (
        "common/AGENTS.md.tpl",
        "codex/agent.toml.tpl",
        "opencode/classic-primary.md.tpl",
        "opencode/classic-subagent.md.tpl",
        "opencode/v2-primary.md.tpl",
        "opencode/v2-subagent.md.tpl",
        "claude/primary.md.tpl",
        "claude/subagent.md.tpl",
        "claude/CLAUDE.md.tpl",
    )
    for template in required_templates:
        if not (AGENT_ROOT / "adapters" / template).is_file():
            errors.append(f"missing adapter template: {template}")

    bootstrap = load_template("common/AGENTS.md.tpl")
    required_npu_prefix = "env -u ASCEND_RT_VISIBLE_DEVICES"
    if required_npu_prefix not in bootstrap:
        errors.append(f"AGENTS bootstrap missing Codex NPU command prefix: {required_npu_prefix}")
    if "sandbox_permissions=require_escalated" not in bootstrap:
        errors.append("AGENTS bootstrap missing Codex NPU escalation requirement")
    if len(bootstrap.encode("utf-8")) > 8192:
        errors.append("AGENTS bootstrap must remain below 8 KiB")

    claude_bootstrap = load_template("claude/CLAUDE.md.tpl")
    if claude_bootstrap.strip() != "@AGENTS.md":
        errors.append("Claude bootstrap must import the shared AGENTS.md")

    if not errors:
        for agent in agents:
            body = (AGENT_ROOT / str(agent["source"])).read_text(encoding="utf-8")
            codex, _ = render_codex(agent, body)
            if tomllib is not None:
                try:
                    parsed = tomllib.loads(codex.decode("utf-8"))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"generated Codex TOML invalid for {agent['name']}: {exc}")
                else:
                    required = {"name", "description", "developer_instructions"}
                    if not required.issubset(parsed):
                        errors.append(f"generated Codex TOML missing required keys for {agent['name']}")
            rendered_adapters = (
                (render_opencode(agent, body, "classic"), "opencode classic"),
                (render_opencode(agent, body, "v2"), "opencode v2"),
                (render_claude(agent, body), "claude"),
            )
            for rendered, tool in rendered_adapters:
                if not rendered.startswith(b"---\n") or b"\n---\n" not in rendered[4:]:
                    errors.append(f"generated {tool} Markdown has invalid frontmatter for {agent['name']}")

    if errors:
        raise InstallError("source validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
    print("[source] manifest, references, skills and adapters are valid")


def check_install(
    layout: TargetLayout,
    source: dict[str, Any],
    opencode_schema: str,
) -> None:
    artifacts = build_artifacts(layout, source, opencode_schema)
    errors: list[str] = []
    manifest = load_install_manifest(layout.manifest_path)
    if manifest is None:
        errors.append(f"missing install manifest: {layout.manifest_path}")
    else:
        try:
            validate_install_manifest_target(manifest, layout, source)
        except InstallError as exc:
            errors.append(str(exc))

    for artifact in artifacts:
        current = current_fingerprint(artifact.path)
        if current != artifact.fingerprint:
            errors.append(f"out of date or missing: {artifact.path}")
    if errors:
        raise InstallError(f"[{layout.tool}] check failed:\n" + "\n".join(f"  - {item}" for item in errors))
    binary = shutil.which(layout.tool)
    binary_note = binary if binary else "not installed; runtime smoke test skipped"
    adapter_note = f"; schema={opencode_schema}" if layout.tool == "opencode" else ""
    print(f"[{layout.tool}] installation is current{adapter_note}; executable: {binary_note}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install TileLang tuning agents for coding tools")
    parser.add_argument("--tool", action="append", choices=SUPPORTED_TOOLS, default=[])
    parser.add_argument("--all-tools", action="store_true", help="select codex, opencode and claude")
    parser.add_argument("--scope", choices=("project", "global"), default="project")
    parser.add_argument(
        "--path",
        help="project agent-entry directory; defaults to the directory containing init.sh",
    )
    parser.add_argument(
        "--opencode-schema",
        choices=("auto", "classic", "v2"),
        default="auto",
        help="OpenCode agent permission schema; auto uses the installed CLI version",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--backup-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-tools", action="store_true")
    args = parser.parse_args(argv)
    if args.backup_existing and args.force:
        parser.error("--backup-existing and --force are mutually exclusive")
    if args.uninstall and args.check:
        parser.error("--uninstall and --check are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.list_tools:
        print(" ".join(SUPPORTED_TOOLS))
        return 0

    source = source_manifest()
    validate_source(source)

    tools = list(SUPPORTED_TOOLS) if args.all_tools else list(dict.fromkeys(args.tool))
    if not tools:
        if args.check:
            return 0
        raise InstallError("no target selected; pass --tool <name> or --all-tools")

    for tool in tools:
        layout = target_layout(tool, args.scope, args.path)
        opencode_schema = opencode_schema_for(
            layout,
            args.opencode_schema,
            preserve_installed=args.check,
        )
        if args.uninstall:
            uninstall(layout, source, args.dry_run)
        elif args.check:
            check_install(layout, source, opencode_schema)
        else:
            install(layout, source, args, opencode_schema)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
