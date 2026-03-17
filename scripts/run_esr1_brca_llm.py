import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_langgraph_workflow import CascadeWorkflow


async def main() -> None:
    wf = CascadeWorkflow()
    result = await wf.run(
        gene="ESR1",
        perturbation_type="knockdown",
        analysis_depth="comprehensive",
        network_source="tcga",
        tcga_network="brca",
        include_llm_insights=True,
    )

    out_dir = Path(__file__).resolve().parents[1] / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "esr1_knockdown_tcga_brca_llm.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())

