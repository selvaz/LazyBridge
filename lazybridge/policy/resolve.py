"""Load profiles and resolve task grants."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .models import EffectivePolicy, PolicyError, PolicyWidening, Profile, ProfileDocument, TaskGrant

TOOL_CLASSES = frozenset({"read", "edit", "shell"})
TOOL_GRANTS = frozenset({"web", "mcp"})
NETWORK_VALUES = frozenset({"on", "off"})
PLACEMENT_VALUES = frozenset({"local", "cloud"})
GRANT_KEYS = frozenset({"tool_classes", "tools", "mcp_servers", "mcp_inventory", "network_routes", "scratch_folder"})
DEFAULT_PROFILE_PATH = Path(__file__).with_name("data") / "profiles.example.toml"


def _expect_keys(value: Mapping[str, Any], allowed: set[str] | frozenset[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PolicyError(f"unknown {where} key(s): {', '.join(unknown)}")


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple | set | frozenset) or any(not isinstance(item, str) for item in value):
        raise PolicyError(f"{where} must be a list of strings")
    return tuple(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def load_profiles(path: Path | str = DEFAULT_PROFILE_PATH) -> ProfileDocument:
    """Load a strictly validated profile document from *path*."""
    try:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise PolicyError(f"cannot load profile document: {exc}") from exc
    _expect_keys(raw, {"version", "revision", "profiles"}, "top-level")
    if not isinstance(raw.get("version"), int) or isinstance(raw.get("version"), bool):
        raise PolicyError("version must be an integer")
    if raw["version"] != 1:
        raise PolicyError(f"unsupported profile schema version: {raw['version']!r}")
    if not isinstance(raw.get("revision"), str) or not raw["revision"]:
        raise PolicyError("revision must be a non-empty string")
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise PolicyError("profiles must be a non-empty table")
    profiles: dict[str, Profile] = {}
    for name, value in raw_profiles.items():
        if not isinstance(value, dict):
            raise PolicyError(f"profile {name!r} must be a table")
        _expect_keys(value, {"allowed_tool_classes"}, f"profile {name!r}")
        classes = frozenset(_strings(value.get("allowed_tool_classes"), f"profile {name!r}.allowed_tool_classes"))
        unknown = classes - TOOL_CLASSES
        if unknown:
            raise PolicyError(f"unknown tool class(es) in profile {name!r}: {', '.join(sorted(unknown))}")
        profiles[name] = Profile(name, classes)
    return ProfileDocument(raw["version"], raw["revision"], MappingProxyType(profiles))


def _normalise_servers(grants: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    requested = grants.get("mcp_servers", {})
    inventory = grants.get("mcp_inventory", requested)
    if not isinstance(requested, dict) or not isinstance(inventory, dict):
        raise PolicyError("mcp_servers and mcp_inventory must be tables")
    if any(not isinstance(name, str) or not name or not isinstance(value, dict) for name, value in inventory.items()):
        raise PolicyError("each mcp_inventory entry must be a named table")
    result: dict[str, Mapping[str, Any]] = {}
    for name, request in requested.items():
        if not isinstance(name, str) or not isinstance(request, dict):
            raise PolicyError("each mcp_servers entry must be a named table")
        _expect_keys(request, {"tools"}, f"MCP grant {name!r}")
        granted_tools = set(_strings(request.get("tools"), f"MCP grant {name!r}.tools"))
        source = inventory.get(name)
        if not isinstance(source, dict):
            raise PolicyError(f"granted MCP server {name!r} is absent from the inventory")
        available_tools = set(_strings(source.get("tools", ()), f"MCP inventory {name!r}.tools"))
        unknown_tools = granted_tools - available_tools
        if unknown_tools:
            raise PolicyError(
                f"MCP server {name!r} does not provide granted tool(s): {', '.join(sorted(unknown_tools))}"
            )
        config = {key: value for key, value in source.items() if key != "tools"}
        config["tools"] = sorted(granted_tools)
        result[name] = _freeze(config)
    return MappingProxyType(result)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve(profile: str, grant: TaskGrant, document: ProfileDocument | None = None) -> EffectivePolicy:
    """Intersect explicit task grants with the named profile ceiling."""
    document = document or load_profiles()
    if profile not in document.profiles:
        raise PolicyError(f"unknown profile: {profile!r}")
    if grant.network not in NETWORK_VALUES:
        raise PolicyError(f"unknown network setting: {grant.network!r}")
    if grant.placement not in PLACEMENT_VALUES:
        raise PolicyError(f"unknown placement setting: {grant.placement!r}")
    if not isinstance(grant.grants, Mapping):
        raise PolicyError("grants must be a mapping")
    _expect_keys(grant.grants, GRANT_KEYS, "grant")
    folder = Path(grant.folder).expanduser().resolve()
    scratch = Path(grant.grants.get("scratch_folder", folder / ".scratch")).expanduser()
    if not scratch.is_absolute():
        scratch = folder / scratch
    scratch = scratch.resolve()
    if scratch != folder and folder not in scratch.parents:
        raise PolicyError("scratch_folder must be inside the assigned folder")
    requested_classes = frozenset(_strings(grant.grants.get("tool_classes", ()), "tool_classes"))
    unknown_classes = requested_classes - TOOL_CLASSES
    if unknown_classes:
        raise PolicyError(f"unknown granted tool class(es): {', '.join(sorted(unknown_classes))}")
    classes = requested_classes & document.profiles[profile].allowed_tool_classes
    tools = frozenset(_strings(grant.grants.get("tools", ()), "tools"))
    unknown_tools = tools - TOOL_GRANTS
    if unknown_tools:
        raise PolicyError(f"unknown tool grant(s): {', '.join(sorted(unknown_tools))}")
    routes = tuple(sorted(set(_strings(grant.grants.get("network_routes", ()), "network_routes"))))
    if grant.network == "off" and routes:
        raise PolicyError("network_routes require network='on'")
    servers = _normalise_servers(grant.grants)
    inventory = tuple(sorted(grant.grants.get("mcp_inventory", grant.grants.get("mcp_servers", {}))))
    if servers and "mcp" not in tools:
        raise PolicyError("MCP servers require the explicit 'mcp' tool grant")
    # Native web and tool hosts are task-controlled egress surfaces.  They are
    # removed, rather than merely annotated, so an engine with incomplete
    # network sandboxing cannot accidentally expose them in an offline task.
    if grant.network == "off":
        tools = tools - {"web", "mcp"}
        servers = MappingProxyType({})
    canonical = {
        "profile": profile,
        "version": document.version,
        "revision": document.revision,
        "folder": str(folder),
        "scratch_folder": str(scratch),
        "tool_classes": sorted(classes),
        "tools": sorted(tools),
        "network": grant.network,
        "network_routes": list(routes),
        "placement": grant.placement,
        "mcp_servers": _thaw(servers),
        "mcp_inventory": inventory,
    }
    return EffectivePolicy(
        profile,
        document.version,
        document.revision,
        folder,
        scratch,
        classes,
        tools,
        grant.network,
        routes,
        grant.placement,
        servers,
        inventory,
        _digest(canonical),
    )



def child_within_parent(
    parent: EffectivePolicy, child_request: TaskGrant,
    profile: str | None = None, document: ProfileDocument | None = None,
) -> EffectivePolicy:
    """Resolve a child and reject every expansion of the immediate parent."""
    profile = profile or parent.profile
    document = document or load_profiles()
    if (document.version, document.revision) != (parent.version, parent.revision):
        raise PolicyWidening("profile document differs from parent")
    child = resolve(profile, child_request, document)
    if child.folder != parent.folder and parent.folder not in child.folder.parents:
        raise PolicyWidening("folder outside parent folder")
    if child.scratch_folder != parent.scratch_folder and parent.scratch_folder not in child.scratch_folder.parents:
        raise PolicyWidening("scratch folder outside parent scratch folder")
    requested_classes = set(_strings(child_request.grants.get("tool_classes", ()), "tool_classes"))
    if not requested_classes <= parent.tool_classes:
        raise PolicyWidening("tool class not granted to parent")
    requested_tools = set(_strings(child_request.grants.get("tools", ()), "tools"))
    if not requested_tools <= parent.tools:
        raise PolicyWidening("tool not granted to parent")
    if child.network == "on" and parent.network == "off":
        raise PolicyWidening("network on under parent off")
    if not set(child.network_routes) <= set(parent.network_routes):
        raise PolicyWidening("network route not granted to parent")
    if child.placement != parent.placement:
        raise PolicyWidening("placement differs from parent")
    if not set(parent.mcp_inventory) <= set(child.mcp_inventory):
        raise PolicyWidening("MCP inventory omits parent server")
    for name, config in child.mcp_servers.items():
        prior = parent.mcp_servers.get(name)
        if prior is None or not set(config["tools"]) <= set(prior["tools"]):
            raise PolicyWidening("MCP server or tool not granted to parent")
        if {k: v for k, v in config.items() if k != "tools"} != {k: v for k, v in prior.items() if k != "tools"}:
            raise PolicyWidening("MCP server configuration differs from parent")
    return child
