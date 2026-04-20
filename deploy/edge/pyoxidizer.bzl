# mind-mem-edge — PyOxidizer build spec (v4.0 prep).
#
# Builds a self-contained `mind-mem-edge` binary with the Python
# runtime + mind-mem embedded. Drops onto on-device agents with no
# pip install step. Pairs with `mind-mem-edge daemon` which proxies
# recall / governance calls up to a central mind-mem cluster.
#
# Build:
#     pyoxidizer build --release
# Output binary:
#     build/<target>/release/install/mind-mem-edge
#
# Development (CPython wheels, fast rebuild):
#     pyoxidizer build
#
# The spec pulls only the core mind-mem modules + markdown backend
# so the edge binary stays under 40 MB. Postgres / ONNX / vector
# deps are opt-in via the `extended` build profile.


def make_exe():
    dist = default_python_distribution(python_version="3.12")
    policy = dist.make_python_packaging_policy()

    # Embed every pure-Python module we can; fall back to filesystem
    # for anything that needs a C extension (sqlite3 is in stdlib, so
    # this is mostly the mind-mem source tree).
    policy.resources_location = "in-memory"
    policy.resources_location_fallback = "filesystem-relative:lib"

    # Keep the binary small — skip anything that's not strictly needed
    # for mind-mem's edge surface.
    policy.bytecode_optimize_level_zero = True

    python_config = dist.make_python_interpreter_config()
    python_config.run_command = (
        "from mind_mem.cli import main; import sys; sys.exit(main(sys.argv[1:]))"
    )

    exe = dist.to_python_executable(
        name="mind-mem-edge",
        packaging_policy=policy,
        config=python_config,
    )

    # mind-mem core. Pulls in block_store, recall, governance, cache,
    # retrieval_trace, feature_gate. Excludes postgres, vector, rerank,
    # training — those are cluster-side.
    exe.add_python_resources(exe.pip_install([
        "mind-mem==3.3.0",
    ]))

    # CLI entry point.
    exe.add_python_resources(
        exe.pip_install(["--no-deps", "mind-mem[cli]==3.3.0"])
    )

    return exe


def make_embedded_resources(exe):
    return exe.to_embedded_resources()


def make_install(exe):
    files = FileManifest()
    files.add_python_resource(".", exe)
    return files


register_target("exe", make_exe)
register_target(
    "resources", make_embedded_resources, depends=["exe"], default_build_script=True
)
register_target(
    "install", make_install, depends=["exe"], default=True
)

resolve_targets()
