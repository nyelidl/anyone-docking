#!/usr/bin/env python3
"""Vendored CLI entrypoint for Meeko's receptor preparation command."""


def _apply_meeko_hydrogen_index_fix() -> bool:
    """Backport the Meeko 0.7.1 stale RDKit hydrogen-index correction."""
    import inspect

    import meeko.polymer as polymer

    source = inspect.getsource(polymer.update_H_positions)
    marker = "to_del = {k: atom.GetIdx() for k, atom in to_del.items()}"
    if marker in source:
        return False

    needle = "    tmpconf = tmpmol.GetConformer()\n"
    if needle not in source:
        raise RuntimeError("Unsupported Meeko update_H_positions implementation")
    source = source.replace(needle, needle + f"    {marker}\n", 1)
    source = source.replace("tmpmol.GetAtomWithIdx(parent.GetIdx())", "tmpmol.GetAtomWithIdx(parent)")
    namespace = dict(polymer.__dict__)
    exec(compile(source, polymer.__file__, "exec"), namespace)
    polymer.update_H_positions = namespace["update_H_positions"]
    return True


def main():
    _apply_meeko_hydrogen_index_fix()
    from meeko.cli.mk_prepare_receptor import main as meeko_main

    return meeko_main()


if __name__ == "__main__":
    raise SystemExit(main())

