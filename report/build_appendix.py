"""Write report/appendix.md: every prompt template, plus one full Tree-of-Thought tree.

    python report/build_appendix.py [path/to/notes_log.json]

Without an argument, the most recent ToT log under data/lectures/*/kits/ is used."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline.prompts import all_templates  # noqa: E402


def latest_tot_log() -> Path | None:
    logs = sorted(config.LECTURES_DIR.glob("*/kits/*/notes_log.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in logs:
        if json.loads(path.read_text(encoding="utf-8")).get("strategy") == "tot":
            return path
    return None


def main():
    parts = ["# Appendix A - Prompt templates\n"]
    for name, text in all_templates().items():
        parts.append(f"## {name}\n\n```text\n{text.strip()}\n```\n")

    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_tot_log()
    if log_path:
        tree = json.loads(log_path.read_text(encoding="utf-8"))["chunks"][0]
        parts.append(f"# Appendix B - One full Tree-of-Thought tree\n\nChunk {tree['chunk_id'] + 1}: "
                     f"*{tree['title']}* - breadth {tree['breadth']}, keep {tree['keep']}, depth {tree['depth']}, "
                     f"{tree['calls']} LLM calls, selected node **{tree['selected']}**.\n")
        for level in tree["levels"]:
            parts.append(f"## Depth {level['depth']}\n")
            for node in level["nodes"]:
                s = node["score"]
                parts.append(
                    f"### Node {node['id']} ({node['style']}, parent {node['parent'] or '-'}) - "
                    f"{'kept' if node.get('kept') else 'pruned'}\n\n"
                    f"Scores: coverage {s.get('coverage')}, faithfulness {s.get('faithfulness')}, clarity "
                    f"{s.get('clarity')}, profile fit {s.get('profile_fit')} -> **{s.get('total')}**  \n"
                    f"Judge: {s.get('reason', '')}  \nFeedback: {node.get('feedback', '')}\n\n"
                    f"```json\n{json.dumps(node['section'], indent=1, ensure_ascii=False)}\n```\n")
    out = Path(__file__).parent / "appendix.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
