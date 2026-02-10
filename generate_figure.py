#!/usr/bin/env python3
"""Generate the CASCADE architecture figure for the JOSS paper."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(10, 7.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.5)
ax.axis('off')

# Color palette
C_CLIENT = '#4A90D9'      # blue
C_SERVER = '#2C3E50'       # dark navy
C_WORKFLOW = '#E8F4FD'     # light blue bg
C_NODE = '#5DADE2'         # workflow nodes
C_ROUTE = '#F39C12'        # routing node (orange)
C_BATCH = '#27AE60'        # batch nodes (green)
C_BATCH_BG = '#EAFAF1'    # light green for parallel block
C_TOOLS = '#8E44AD'        # tool modules (purple)
C_EXTERNAL = '#E74C3C'     # external APIs (red)
C_REPORT = '#1ABC9C'       # report (teal)
C_TEXT = '#FFFFFF'
C_DARK = '#2C3E50'
C_DATA = '#34495E'         # pre-computed data (dark)

def box(x, y, w, h, color, label, fontsize=8, textcolor='white', alpha=1.0,
        style='round,pad=0.1', edgecolor='#34495E', linestyle='solid', lw=1.2):
    fancy = FancyBboxPatch((x, y), w, h, boxstyle=style,
                           facecolor=color, edgecolor=edgecolor,
                           linewidth=lw, alpha=alpha, zorder=2,
                           linestyle=linestyle)
    ax.add_patch(fancy)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=textcolor, zorder=3)

def arrow(x1, y1, x2, y2, color='#7F8C8D', style='->', lw=1.5, linestyle='solid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                shrinkA=2, shrinkB=2, linestyle=linestyle),
                zorder=1)

# === Row 1: MCP Client ===
box(3, 6.7, 4, 0.5, C_CLIENT, 'MCP Client', fontsize=9)

# Arrow down with clear arrowhead
arrow(5, 6.7, 5, 6.4, color=C_DARK, style='-|>', lw=2.0)

# === Row 2: MCP Server outer box ===
server_bg = FancyBboxPatch((0.3, 0.3), 9.4, 6.1, boxstyle='round,pad=0.15',
                           facecolor='#F8F9FA', edgecolor=C_SERVER,
                           linewidth=2, alpha=0.9, zorder=0)
ax.add_patch(server_bg)
ax.text(5, 6.25, 'CASCADE LangGraph MCP Server', ha='center', va='center',
        fontsize=11, fontweight='bold', color=C_DARK, zorder=3)

# === Row 3: Workflow box ===
wf_bg = FancyBboxPatch((0.6, 1.9), 8.8, 4.1, boxstyle='round,pad=0.1',
                        facecolor=C_WORKFLOW, edgecolor='#85C1E9',
                        linewidth=1.5, alpha=0.7, zorder=1)
ax.add_patch(wf_bg)
ax.text(5, 5.8, 'LangGraph StateGraph Workflow', ha='center', va='center',
        fontsize=9, fontweight='bold', color='#2980B9', zorder=3)

# --- Sequential init nodes ---
box(0.9, 5.1, 1.5, 0.45, C_NODE, 'Initialize', fontsize=7.5)
box(2.7, 5.1, 1.7, 0.45, C_NODE, 'Resolve Gene', fontsize=7.5)
box(4.7, 5.1, 2.0, 0.45, C_NODE, 'Classify Role', fontsize=7.5)

arrow(2.4, 5.32, 2.7, 5.32)
arrow(4.4, 5.32, 4.7, 5.32)

# --- Routing node ---
box(7.0, 5.1, 1.8, 0.45, C_ROUTE, 'Route', fontsize=8, textcolor='white')
arrow(6.7, 5.32, 7.0, 5.32)

# --- Parallel execution background block ---
parallel_bg = FancyBboxPatch((0.7, 3.95), 8.6, 1.05, boxstyle='round,pad=0.08',
                              facecolor=C_BATCH_BG, edgecolor='#27AE60',
                              linewidth=1.0, alpha=0.5, zorder=1,
                              linestyle='dashed')
ax.add_patch(parallel_bg)
# "Parallel Execution" label at top-left of the parallel block
ax.text(0.85, 4.9, 'Parallel Execution', ha='left', va='center',
        fontsize=6.5, fontstyle='italic', color='#1E8449', zorder=3,
        bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                  edgecolor='#27AE60', linewidth=0.5, alpha=0.9))

# --- Three parallel batch nodes ---
# Batch Core
box(0.9, 4.15, 2.6, 0.65, C_BATCH, 'Batch Core\nPerturbation | Regulators | Targets', fontsize=6.5)

# Batch External
box(3.7, 4.15, 2.6, 0.65, C_BATCH, 'Batch External\nSTRING PPI | LINCS | Super-Enh', fontsize=6.5)

# Batch Insights
box(6.5, 4.15, 2.6, 0.65, C_BATCH, 'Batch Insights\nSimilarity | Vulnerability | Cross-Cell', fontsize=6.5)

# Fork arrows from router to batches
arrow(7.9, 5.1, 2.2, 4.8, color='#F39C12', lw=1.5)
arrow(7.9, 5.1, 5.0, 4.8, color='#F39C12', lw=1.5)
arrow(7.9, 5.1, 7.8, 4.8, color='#F39C12', lw=1.5)

# --- Report + Synthesis row ---
box(2.5, 3.05, 2.3, 0.5, C_REPORT, 'Generate Report', fontsize=8)
box(5.2, 3.05, 2.3, 0.5, '#16A085', 'LLM Synthesis\n(optional)', fontsize=7, textcolor='white')

# Join arrows from batches to report
arrow(2.2, 4.15, 3.65, 3.55, color='#27AE60', lw=1.5)
arrow(5.0, 4.15, 3.65, 3.55, color='#27AE60', lw=1.5)
arrow(7.8, 4.15, 3.65, 3.55, color='#27AE60', lw=1.5)

arrow(4.8, 3.3, 5.2, 3.3)

# === Row 4: Data sources ===
# Local Analysis Modules (dashed border = local)
box(0.5, 0.5, 2.5, 1.1, C_TOOLS,
    'Local Analysis Modules\nNetwork Propagation\nGene Embeddings\nSimilarity Cache\nGene ID Resolution',
    fontsize=6.5, linestyle='dashed', edgecolor=C_TOOLS)

# External / Runtime APIs (solid border + colored edge = external)
box(3.8, 0.5, 2.5, 1.1, C_EXTERNAL,
    'Runtime APIs\nSTRING DB\nLINCS L1000\ndbSUPER\nEnsembl API',
    fontsize=6.5, edgecolor='#C0392B', lw=1.8)

# Pre-computed Data (dashed border = local)
box(7.0, 0.5, 2.5, 1.1, C_DATA,
    'Pre-computed Data\n10 Regulatory Networks\nGREmLN Checkpoint\nGene ID Cache',
    fontsize=6.5, linestyle='dashed', edgecolor=C_DATA)

# === Data flow arrows: sources UP to batch nodes ===
# Local Analysis Modules (purple) -> Batch Core (uses propagation, embeddings)
arrow(1.75, 1.6, 2.2, 4.15, color=C_TOOLS, style='->', lw=1.5, linestyle='dashed')

# Runtime APIs (red) -> Batch External (STRING, LINCS, dbSUPER)
arrow(5.05, 1.6, 5.0, 4.15, color=C_EXTERNAL, style='->', lw=1.5)

# Pre-computed Data (dark) -> Batch Core (networks used in propagation)
arrow(8.0, 1.6, 2.5, 4.15, color=C_DATA, style='->', lw=1.5, linestyle='dashed')

# Pre-computed Data (dark) -> Batch Insights (checkpoint used in similarity)
arrow(8.5, 1.6, 7.8, 4.15, color=C_DATA, style='->', lw=1.5, linestyle='dashed')

# === Legend ===
legend_items = [
    mpatches.Patch(color=C_NODE, label='Sequential nodes'),
    mpatches.Patch(color=C_ROUTE, label='Conditional routing'),
    mpatches.Patch(color=C_BATCH, label='Parallel batch execution'),
    mpatches.Patch(color=C_REPORT, label='Report synthesis'),
]

# Second legend row for data source types
from matplotlib.lines import Line2D
data_legend = [
    Line2D([0], [0], color=C_TOOLS, lw=1.5, linestyle='dashed', label='Local data flow'),
    Line2D([0], [0], color=C_EXTERNAL, lw=1.5, linestyle='solid', label='Runtime API call'),
]

leg1 = ax.legend(handles=legend_items, loc='lower center', ncol=4, fontsize=7,
                 frameon=True, fancybox=True, framealpha=0.9,
                 bbox_to_anchor=(0.42, -0.02))
ax.add_artist(leg1)
ax.legend(handles=data_legend, loc='lower center', ncol=2, fontsize=7,
          frameon=True, fancybox=True, framealpha=0.9,
          bbox_to_anchor=(0.85, -0.02))

plt.tight_layout()
plt.savefig('figure_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figure_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved figure_architecture.png and figure_architecture.pdf")
