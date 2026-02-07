#!/usr/bin/env python3
"""Generate the CASCADE architecture figure for the JOSS paper."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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
C_TOOLS = '#8E44AD'        # tool modules (purple)
C_EXTERNAL = '#E74C3C'     # external APIs (red)
C_REPORT = '#1ABC9C'       # report (teal)
C_TEXT = '#FFFFFF'
C_DARK = '#2C3E50'

def box(x, y, w, h, color, label, fontsize=8, textcolor='white', alpha=1.0, style='round,pad=0.1'):
    fancy = FancyBboxPatch((x, y), w, h, boxstyle=style,
                           facecolor=color, edgecolor='#34495E',
                           linewidth=1.2, alpha=alpha, zorder=2)
    ax.add_patch(fancy)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=textcolor, zorder=3)

def arrow(x1, y1, x2, y2, color='#7F8C8D', style='->', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=2, shrinkB=2),
                zorder=1)

# === Row 1: MCP Client ===
box(3, 6.7, 4, 0.5, C_CLIENT, 'MCP Client', fontsize=9)

# Arrow down
arrow(5, 6.7, 5, 6.4)

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

# --- Three parallel batch rows ---
# Batch Core
box(0.9, 4.15, 2.6, 0.65, C_BATCH, 'Batch Core\nPerturbation | Regulators | Targets', fontsize=6.5)

# Batch External
box(3.7, 4.15, 2.6, 0.65, C_BATCH, 'Batch External\nSTRING PPI | LINCS | Super-Enh', fontsize=6.5)

# Batch Insights
box(6.5, 4.15, 2.6, 0.65, C_BATCH, 'Batch Insights\nSimilarity | Vulnerability | Cross-Cell', fontsize=6.5)

# Arrows from router to batches
arrow(7.9, 5.1, 2.2, 4.8, color='#F39C12')
arrow(7.9, 5.1, 5.0, 4.8, color='#F39C12')
arrow(7.9, 5.1, 7.8, 4.8, color='#F39C12')

# --- Report + Synthesis row ---
box(2.5, 3.05, 2.3, 0.5, C_REPORT, 'Generate Report', fontsize=8)
box(5.2, 3.05, 2.3, 0.5, '#16A085', 'LLM Synthesis\n(optional)', fontsize=7, textcolor='white')

# Arrows from batches to report
arrow(2.2, 4.15, 3.65, 3.55, color='#27AE60')
arrow(5.0, 4.15, 3.65, 3.55, color='#27AE60')
arrow(7.8, 4.15, 3.65, 3.55, color='#27AE60')

arrow(4.8, 3.3, 5.2, 3.3)

# === Row 4: Data sources ===
# Tool modules
box(0.5, 0.5, 2.5, 1.1, C_TOOLS, 'Core Tools\nNetwork Propagation\nGene Embeddings\nSimilarity Cache\nGene ID Resolution', fontsize=6.5)

# External APIs
box(3.8, 0.5, 2.5, 1.1, C_EXTERNAL, 'External Data\nSTRING DB\nLINCS L1000\ndbSUPER\nEnsembl API', fontsize=6.5)

# Data files
box(7.0, 0.5, 2.5, 1.1, '#34495E', 'Data Assets\n10 Regulatory Networks\nGREmLN Checkpoint\nGene ID Cache', fontsize=6.5)

# Dashed arrows from workflow to data sources
arrow(2.2, 3.05, 1.75, 1.6, color='#8E44AD', style='->', lw=1.0)
arrow(5.0, 3.05, 5.05, 1.6, color='#E74C3C', style='->', lw=1.0)
arrow(7.8, 3.05, 8.25, 1.6, color='#7F8C8D', style='->', lw=1.0)

# === Legend ===
legend_items = [
    mpatches.Patch(color=C_NODE, label='Sequential nodes'),
    mpatches.Patch(color=C_ROUTE, label='Conditional routing'),
    mpatches.Patch(color=C_BATCH, label='Parallel batch execution'),
    mpatches.Patch(color=C_REPORT, label='Report synthesis'),
]
ax.legend(handles=legend_items, loc='lower center', ncol=4, fontsize=8,
          frameon=True, fancybox=True, framealpha=0.9,
          bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig('figure_architecture.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figure_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved figure_architecture.png and figure_architecture.pdf")
