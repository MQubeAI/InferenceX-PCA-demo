"""Streaming PostgreSQL ingestion for candidate InferenceX snapshots.

No function in this module materialises the decompressed dump.  The archive is
read from verified Zstandard parts, decompressed into a pipe, and consumed by
``pg_restore``.  A caller must provide an isolated PostgreSQL 17 database URL
or start one explicitly through the candidate script.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import socket
import subprocess
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from modeling.snapshot_refresh import OfficialRelease, SnapshotRefreshError


PART_NAME = re.compile(r"^inferencex-\d{4}-\d{2}-\d{2}\.dump\.zst\.part\d+$")
SHA256_LINE = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")
LOGICAL_TABLES = ("benchmark_results", "configs")


def parse_sha256sums(text: str) -> dict[str, str]:
    """Parse GNU-style upstream checksum lines and reject ambiguous entries."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        match = SHA256_LINE.fullmatch(line)
        if not match:
            raise SnapshotRefreshError(f"Invalid SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        if Path(name).name != name or name in values:
            raise SnapshotRefreshError(f"Unsafe or duplicate SHA256SUMS asset: {name}")
        values[name] = digest.lower()
    if not values:
        raise SnapshotRefreshError("SHA256SUMS does not contain any checksums.")
    return values


def upstream_dump_assets(release: OfficialRelease) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the SHA256SUMS asset and ordered dump-part assets for an official release."""

    by_name = {asset.get("name"): asset for asset in release.assets if isinstance(asset.get("name"), str)}
    checksums = by_name.get("SHA256SUMS")
    parts = [asset for name, asset in by_name.items() if isinstance(name, str) and PART_NAME.fullmatch(name)]
    if not checksums or not parts:
        raise SnapshotRefreshError(
            f"Official release {release.tag} must contain SHA256SUMS and dump.zst.part assets."
        )
    if any(not isinstance(asset.get("browser_download_url"), str) for asset in [checksums, *parts]):
        raise SnapshotRefreshError(f"Official release {release.tag} has an asset without a download URL.")
    return checksums, tuple(sorted(parts, key=lambda asset: asset["name"]))  # type: ignore[return-value]


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "InferenceX-refresh"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    except OSError as exc:
        raise SnapshotRefreshError(f"Upstream asset download failed: {exc}") from exc


def download_and_verify_upstream_parts(release: OfficialRelease, destination: str | Path) -> list[Path]:
    """Download every official part and validate it before any restore begins."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    checksum_asset, parts = upstream_dump_assets(release)
    checksums_path = root / "SHA256SUMS"
    _download(checksum_asset["browser_download_url"], checksums_path)
    checksums = parse_sha256sums(checksums_path.read_text(encoding="utf-8"))
    paths: list[Path] = []
    for asset in parts:
        name = asset["name"]
        if name not in checksums:
            raise SnapshotRefreshError(f"Official SHA256SUMS is missing {name}.")
        path = root / name
        _download(asset["browser_download_url"], path)
        actual = sha256_file(path)
        if actual != checksums[name]:
            raise SnapshotRefreshError(f"Upstream checksum mismatch for {name}.")
        paths.append(path)
    return paths


def verify_upstream_parts(release: OfficialRelease, directory: str | Path) -> list[Path]:
    """Re-verify a pre-downloaded official release without trusting its presence.

    This is used by the replay fixture to avoid a second multi-gigabyte download
    after a verified retrieval.  It performs the identical SHA-256 gate before
    streaming and never accepts an arbitrary local dump directory.
    """

    root = Path(directory)
    checksum_asset, parts = upstream_dump_assets(release)
    checksum_path = root / checksum_asset["name"]
    if not checksum_path.is_file():
        raise SnapshotRefreshError("Pre-downloaded upstream directory lacks SHA256SUMS.")
    checksums = parse_sha256sums(checksum_path.read_text(encoding="utf-8"))
    verified: list[Path] = []
    for asset in parts:
        name = asset["name"]
        path = root / name
        if name not in checksums or not path.is_file():
            raise SnapshotRefreshError(f"Pre-downloaded upstream directory lacks verified asset {name}.")
        if sha256_file(path) != checksums[name]:
            raise SnapshotRefreshError(f"Upstream checksum mismatch for {name}.")
        verified.append(path)
    return verified


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_free_disk(path: str | Path, minimum_gib: int = 60) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    required = minimum_gib * 1024**3
    if usage.free < required:
        raise SnapshotRefreshError(
            f"Candidate ingestion requires at least {minimum_gib} GiB free; found {usage.free / 1024**3:.1f} GiB."
        )
    return {"free_bytes": usage.free, "total_bytes": usage.total, "minimum_required_bytes": required}


def postgres_major(pg_restore: str = "pg_restore") -> int:
    try:
        output = subprocess.check_output([pg_restore, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SnapshotRefreshError("pg_restore is required for candidate ingestion.") from exc
    match = re.search(r"(\d+)(?:\.\d+)?", output)
    if not match:
        raise SnapshotRefreshError(f"Could not determine pg_restore version: {output.strip()}")
    return int(match.group(1))


def require_postgres17(pg_restore: str = "pg_restore") -> None:
    major = postgres_major(pg_restore)
    if major != 17:
        raise SnapshotRefreshError(f"Candidate ingestion requires PostgreSQL 17 pg_restore; found {major}.")


def _postgres_tool(name: str) -> str:
    """Locate a PostgreSQL server tool, preferring pg_config's matching bindir."""

    pg_config = shutil.which("pg_config")
    if pg_config:
        try:
            bindir = subprocess.check_output([pg_config, "--bindir"], text=True).strip()
            candidate = Path(bindir) / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        except (OSError, subprocess.CalledProcessError):
            pass
    resolved = shutil.which(name)
    if not resolved:
        raise SnapshotRefreshError(f"PostgreSQL 17 server tool is required: {name}.")
    return resolved


def _server_major(initdb: str) -> int:
    try:
        output = subprocess.check_output([initdb, "--version"], text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SnapshotRefreshError("initdb is required for an isolated PostgreSQL candidate cluster.") from exc
    match = re.search(r"(\d+)(?:\.\d+)?", output)
    if not match:
        raise SnapshotRefreshError(f"Could not determine initdb version: {output.strip()}")
    return int(match.group(1))


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@contextmanager
def isolated_postgres17(workdir: str | Path, *, database_name: str) -> Iterator[dict[str, Any]]:
    """Start a private PostgreSQL 17 cluster and always remove it afterwards.

    The cluster is under the caller's scratch directory, listens only on
    loopback, and is never a developer's existing PostgreSQL service.  It is a
    deliberately non-Docker option for the heavy refresh runner.
    """

    if not re.fullmatch(r"inferencex_[a-z0-9_]+", database_name):
        raise SnapshotRefreshError("Temporary database name must be explicitly inferencex-isolated.")
    require_postgres17()
    initdb, pg_ctl, createdb = (_postgres_tool(name) for name in ("initdb", "pg_ctl", "createdb"))
    if _server_major(initdb) != 17:
        raise SnapshotRefreshError(f"Candidate ingestion requires PostgreSQL 17 initdb; found {_server_major(initdb)}.")
    root = Path(workdir)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".inferencex-pg17-", dir=root))
    data_dir, log_path = temporary / "data", temporary / "postgres.log"
    port = _unused_loopback_port()
    started = False
    try:
        try:
            subprocess.run(
                [initdb, "-D", str(data_dir), "--no-locale", "--encoding=UTF8", "-U", "inferencex"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [pg_ctl, "-D", str(data_dir), "-l", str(log_path), "-o", f"-h 127.0.0.1 -p {port}", "start"],
                check=True,
                capture_output=True,
                text=True,
            )
            started = True
            subprocess.run(
                [createdb, "-h", "127.0.0.1", "-p", str(port), "-U", "inferencex", database_name],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            raise SnapshotRefreshError(f"Could not initialize isolated PostgreSQL 17: {detail}") from exc
        yield {
            "database_url": f"postgresql://inferencex@127.0.0.1:{port}/{database_name}",
            "data_directory": str(data_dir),
            "port": port,
            "postgres_version": 17,
        }
    finally:
        if started:
            subprocess.run([pg_ctl, "-D", str(data_dir), "-m", "fast", "stop"], check=False, capture_output=True, text=True)
        shutil.rmtree(temporary, ignore_errors=True)


def _stream_pg_restore(
    parts: Iterable[Path],
    pg_restore_args: list[str],
    *,
    zstd: str = "zstd",
    pg_restore: str = "pg_restore",
) -> tuple[str, str]:
    """Feed concatenated parts through zstd to pg_restore without an archive file."""

    try:
        decompressor = subprocess.Popen(
            [zstd, "--long=27", "-d", "-c"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        restore = subprocess.Popen(
            [pg_restore, *pg_restore_args],
            stdin=decompressor.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
    except OSError as exc:
        raise SnapshotRefreshError("zstd and pg_restore must be installed for candidate ingestion.") from exc
    assert decompressor.stdin is not None and decompressor.stdout is not None
    decompressor.stdout.close()
    upstream_closed_early = False
    try:
        for part in parts:
            with part.open("rb") as handle:
                shutil.copyfileobj(handle, decompressor.stdin, length=1024 * 1024)
    except BrokenPipeError:
        # ``pg_restore --list`` reads the custom-format TOC near the start of
        # the stream and exits successfully without consuming table payloads.
        # A selective restore may do the same after its final selected entry.
        upstream_closed_early = True
    finally:
        try:
            decompressor.stdin.close()
        except BrokenPipeError:
            upstream_closed_early = True
    restore_stdout, restore_stderr = restore.communicate()
    decompressor_stderr = decompressor.stderr.read() if decompressor.stderr else b""
    decompressor_status = decompressor.wait()
    expected_sigpipe = decompressor_status in (-13, 141) or b"Broken pipe" in decompressor_stderr
    if decompressor_status != 0 and not (upstream_closed_early and restore.returncode == 0 and expected_sigpipe):
        raise SnapshotRefreshError("zstd decompression failed: " + decompressor_stderr.decode("utf-8", "replace"))
    if restore.returncode != 0:
        raise SnapshotRefreshError("pg_restore failed: " + restore_stderr.decode("utf-8", "replace"))
    return restore_stdout.decode("utf-8", "replace"), restore_stderr.decode("utf-8", "replace")


def list_archive_toc(parts: Iterable[Path], *, zstd: str = "zstd", pg_restore: str = "pg_restore") -> list[str]:
    output, _stderr = _stream_pg_restore(parts, ["--list"], zstd=zstd, pg_restore=pg_restore)
    lines = [line for line in output.splitlines() if line and not line.startswith(";")]
    if not lines:
        raise SnapshotRefreshError("pg_restore TOC is empty or unreadable.")
    return lines


def minimal_restore_toc(
    toc_lines: Iterable[str],
    tables: Iterable[str] = LOGICAL_TABLES,
    *,
    dependency_names: Iterable[str] = (),
) -> list[str]:
    """Select required tables plus explicitly observed TOC dependencies only.

    Table dependencies are deliberately *not* restored wholesale.  If a first
    restore identifies a missing user type/function/domain, the caller passes
    that exact name back here and retries the isolated candidate database.
    """

    selected: list[str] = []
    names = tuple(tables)
    dependencies = {name.rsplit(".", 1)[-1] for name in dependency_names}
    allowed = ("SCHEMA", "TABLE", "TABLE DATA", "SEQUENCE", "SEQUENCE SET", "DEFAULT", "CONSTRAINT", "INDEX")
    for line in toc_lines:
        upper = line.upper()
        target_table = any(re.search(rf"\b{re.escape(name)}(?:\b|_)", line) for name in names)
        explicit_dependency = (
            any(f" {kind} " in upper for kind in ("TYPE", "DOMAIN", "FUNCTION", "EXTENSION"))
            and any(re.search(rf"\b{re.escape(name)}\b", line) for name in dependencies)
        )
        foreign_key_to_unrestored_table = " FK CONSTRAINT " in upper
        if not foreign_key_to_unrestored_table and (
            (any(f" {kind} " in upper for kind in allowed) and (" SCHEMA " in upper or target_table))
            or explicit_dependency
        ):
            selected.append(line)
    if not any(" TABLE " in line.upper() and "benchmark_results" in line for line in selected):
        raise SnapshotRefreshError("TOC does not contain benchmark_results table metadata.")
    if not any(" TABLE " in line.upper() and re.search(r"\bconfigs\b", line) for line in selected):
        raise SnapshotRefreshError("TOC does not contain configs table metadata.")
    return selected


MISSING_DEPENDENCY = re.compile(r"(?:type|domain|function|extension) \"([^\"]+)\" does not exist", re.IGNORECASE)


def _missing_dependency_names(error_text: str) -> set[str]:
    return {match.group(1).rsplit(".", 1)[-1] for match in MISSING_DEPENDENCY.finditer(error_text)}


def _reset_isolated_database(database_url: str, *, psql: str = "psql") -> None:
    """Remove only user schemas inside the clearly isolated candidate database."""

    if "inferencex" not in database_url.lower():
        raise SnapshotRefreshError("Refusing to reset a database that is not explicitly inferencex-isolated.")
    schemas_sql = (
        "SELECT nspname FROM pg_namespace "
        "WHERE nspname NOT LIKE 'pg_%' AND nspname <> 'information_schema' ORDER BY nspname;"
    )
    schemas = [line.strip() for line in _psql(database_url, schemas_sql, psql=psql).splitlines() if line.strip()]
    for schema in schemas:
        _psql(database_url, f"DROP SCHEMA {_quote_identifier(schema)} CASCADE;", psql=psql)
    _psql(database_url, "CREATE SCHEMA IF NOT EXISTS public;", psql=psql)


def selective_stream_restore(
    parts: list[Path],
    *,
    database_url: str,
    workdir: str | Path,
    zstd: str = "zstd",
    pg_restore: str = "pg_restore",
) -> dict[str, Any]:
    """Restore only the TOC-selected table/type dependencies into one isolated DB."""

    require_postgres17(pg_restore)
    toc = list_archive_toc(parts, zstd=zstd, pg_restore=pg_restore)
    dependencies: set[str] = set()
    list_path = Path(workdir) / "minimal-restore.list"
    for attempt in range(1, 9):
        selected = minimal_restore_toc(toc, dependency_names=dependencies)
        list_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
        try:
            _stream_pg_restore(
                parts,
                [
                    "--exit-on-error",
                    "--no-owner",
                    "--no-privileges",
                    "--dbname",
                    database_url,
                    "--use-list",
                    str(list_path),
                ],
                zstd=zstd,
                pg_restore=pg_restore,
            )
            return {
                "toc_entries": len(toc),
                "selected_entries": len(selected),
                "selected_dependencies": sorted(dependencies),
                "attempts": attempt,
                "toc_list": str(list_path),
            }
        except SnapshotRefreshError as exc:
            missing = _missing_dependency_names(str(exc)) - dependencies
            if not missing:
                raise
            dependency_lines = minimal_restore_toc(toc, dependency_names=dependencies | missing)
            if len(dependency_lines) == len(selected):
                raise SnapshotRefreshError(
                    f"Selective restore reported missing dependency {sorted(missing)} but no matching TOC object was found. "
                    "Refusing a full-database restore."
                ) from exc
            dependencies |= missing
            _reset_isolated_database(database_url)
    raise SnapshotRefreshError("Selective restore exceeded the minimal dependency retry limit.")


def _quote_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise SnapshotRefreshError(f"Unsafe PostgreSQL identifier: {value!r}")
    return '"' + value + '"'


def _psql(database_url: str, sql: str, *, psql: str = "psql", output: Path | None = None) -> str:
    command = [psql, "--no-psqlrc", "--set", "ON_ERROR_STOP=1", "--dbname", database_url, "--tuples-only", "--no-align", "--command", sql]
    try:
        if output is None:
            result = subprocess.run(command, check=True, text=True, capture_output=True)
            return result.stdout
        with output.open("w", encoding="utf-8", newline="") as handle:
            subprocess.run(command, check=True, text=True, stdout=handle, stderr=subprocess.PIPE)
            return ""
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise SnapshotRefreshError(f"psql query failed: {detail}") from exc


def discover_restored_tables(database_url: str, *, psql: str = "psql") -> dict[str, tuple[str, str]]:
    """Discover schema/table names after restore instead of assuming public."""

    query = "SELECT table_schema || E'\\t' || table_name FROM information_schema.tables WHERE table_type='BASE TABLE' ORDER BY table_schema, table_name;"
    rows = [line.split("\t", 1) for line in _psql(database_url, query, psql=psql).splitlines() if "\t" in line]
    matches: dict[str, tuple[str, str]] = {}
    for logical in LOGICAL_TABLES:
        found = [(schema, table) for schema, table in rows if table == logical]
        if len(found) != 1:
            raise SnapshotRefreshError(f"Expected exactly one restored {logical} table; found {found}.")
        matches[logical] = found[0]
    return matches


def export_deterministic_tables(
    database_url: str,
    destination: str | Path,
    *,
    psql: str = "psql",
) -> dict[str, Any]:
    """COPY tables ordered by their stable primary id to canonical raw CSVs."""

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    tables = discover_restored_tables(database_url, psql=psql)
    outputs: dict[str, Path] = {}
    for logical, filename in (("benchmark_results", "benchmark_results_raw.csv"), ("configs", "configs.csv")):
        schema, table = tables[logical]
        quoted = f"{_quote_identifier(schema)}.{_quote_identifier(table)}"
        # Both required source tables use immutable primary ``id`` keys.  Refuse
        # any future schema that cannot provide that deterministic ordering.
        column_query = (
            "SELECT a.attname FROM pg_index i JOIN pg_attribute a ON a.attrelid=i.indrelid "
            "AND a.attnum=ANY(i.indkey) WHERE i.indrelid='"
            + schema + "." + table
            + "'::regclass AND i.indisprimary ORDER BY array_position(i.indkey, a.attnum);"
        )
        primary_key = [line.strip() for line in _psql(database_url, column_query, psql=psql).splitlines() if line.strip()]
        if primary_key != ["id"]:
            raise SnapshotRefreshError(f"{schema}.{table} must have primary key id for deterministic export; found {primary_key}.")
        copy_sql = f"COPY (SELECT * FROM {quoted} ORDER BY \"id\") TO STDOUT WITH (FORMAT CSV, HEADER true, ENCODING 'UTF8');"
        output = root / filename
        _psql(database_url, copy_sql, psql=psql, output=output)
        outputs[filename] = output
    return {
        "tables": {name: {"schema": schema, "table": table} for name, (schema, table) in tables.items()},
        "files": {name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for name, path in outputs.items()},
    }
