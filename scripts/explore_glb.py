"""Quick exploration script to inspect the internal structure of GLB assets.

Usage:
    python scripts/explore_glb.py                       # summarize all assets
    python scripts/explore_glb.py "00686245121504.glb"  # one file in detail
"""
import sys
from pathlib import Path

import trimesh

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def explore(path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"File: {path.name}  ({path.stat().st_size / 1024:.1f} KB)")
    print(f"{'='*60}")

    obj = trimesh.load(str(path), process=False)
    print(f"Type: {type(obj).__name__}")

    if isinstance(obj, trimesh.Scene):
        print(f"Geometries: {len(obj.geometry)}")
        print(f"Graph nodes: {len(obj.graph.nodes)}")

        for name, geo in obj.geometry.items():
            print(f"\n  Mesh: {name!r}")
            print(f"    Vertices: {len(geo.vertices)}")
            print(f"    Faces:    {len(geo.faces)}")
            print(f"    Bounds:   {geo.bounds.tolist()}")
            print(f"    Visual type: {type(geo.visual).__name__}")
            if hasattr(geo.visual, "material") and geo.visual.material is not None:
                mat = geo.visual.material
                print(f"    Material: {type(mat).__name__}")
                if hasattr(mat, "baseColorFactor"):
                    print(f"      baseColorFactor: {mat.baseColorFactor}")
                if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
                    tex = mat.baseColorTexture
                    print(f"      Texture size: {tex.size if hasattr(tex, 'size') else 'N/A'}")
    elif isinstance(obj, trimesh.Trimesh):
        print(f"  Vertices: {len(obj.vertices)}")
        print(f"  Faces:    {len(obj.faces)}")
        print(f"  Bounds:   {obj.bounds.tolist()}")
    else:
        print(f"  Unexpected type: {type(obj)}")


def main():
    glbs = sorted(ASSETS_DIR.glob("*.glb"))
    if not glbs:
        print("No .glb files found in assets/")
        sys.exit(1)

    # Explore first file in detail, then summarize the rest
    if len(sys.argv) > 1:
        targets = [ASSETS_DIR / sys.argv[1]]
    else:
        targets = glbs

    for path in targets:
        explore(path)


if __name__ == "__main__":
    main()
