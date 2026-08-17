#!/usr/bin/env python3
"""
Export Batch Generation Statistics table from LaTeX to Excel
"""

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# Data from the Batch Generation Statistics table
data = {
    'Metric': [
        'Total Generated',
        'Valid SMILES',
        'Parseable Molecules',
        '',  # Separator row
        'CRAM Compliance:',
        '≥2 Carboxyl Groups',
        '≥70% Aliphatic Carbon',
        'O/C Ratio (0.4-0.6)',
        '≥1 Alicyclic Ring',
        '',  # Separator row
        'Full CRAM Criteria'
    ],
    'Count': [
        10000,
        9721,
        9683,
        '',
        '',
        7842,
        8156,
        6924,
        7381,
        '',
        3921
    ],
    'Percentage': [
        '100.0%',
        '97.2%',
        '96.8%',
        '',
        '',
        '78.4%',
        '81.6%',
        '69.2%',
        '73.8%',
        '',
        '39.2%'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel with formatting
output_file = 'Batch_Generation_Statistics.xlsx'

# Create workbook and worksheet
wb = Workbook()
ws = wb.active
ws.title = "CRAM Generation Performance"

# Define styles
header_font = Font(bold=True, size=12)
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
header_alignment = Alignment(horizontal="center", vertical="center")

section_font = Font(bold=True, size=11)
section_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")

highlight_font = Font(bold=True, size=11)
highlight_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)

# Add title
ws.merge_cells('A1:C1')
ws['A1'] = "CRAM Generation Performance"
ws['A1'].font = Font(bold=True, size=14)
ws['A1'].alignment = Alignment(horizontal="center", vertical="center")
ws['A1'].fill = PatternFill(start_color="2F5496", end_color="2F5496", fill_type="solid")
ws['A1'].font = Font(bold=True, size=14, color="FFFFFF")
ws.row_dimensions[1].height = 25

# Add headers
headers = ['Metric', 'Count', 'Percentage']
for col_idx, header in enumerate(headers, start=1):
    cell = ws.cell(row=2, column=col_idx, value=header)
    cell.font = header_font
    cell.fill = header_fill
    cell.alignment = header_alignment
    cell.border = border
    cell.font = Font(bold=True, size=11, color="FFFFFF")

ws.row_dimensions[2].height = 20

# Add data
for row_idx, (metric, count, percentage) in enumerate(zip(df['Metric'], df['Count'], df['Percentage']), start=3):
    # Metric column
    cell_metric = ws.cell(row=row_idx, column=1, value=metric)
    cell_metric.border = border
    
    # Count column
    cell_count = ws.cell(row=row_idx, column=2, value=count if count != '' else '')
    cell_count.border = border
    if count != '':
        cell_count.alignment = Alignment(horizontal="right")
    
    # Percentage column
    cell_percentage = ws.cell(row=row_idx, column=3, value=percentage)
    cell_percentage.border = border
    cell_percentage.alignment = Alignment(horizontal="center")
    
    # Apply special formatting
    if metric == 'CRAM Compliance:':
        cell_metric.font = section_font
        cell_metric.fill = section_fill
        cell_count.fill = section_fill
        cell_percentage.fill = section_fill
    elif metric == 'Full CRAM Criteria':
        cell_metric.font = highlight_font
        cell_metric.fill = highlight_fill
        cell_count.font = highlight_font
        cell_count.fill = highlight_fill
        cell_percentage.font = highlight_font
        cell_percentage.fill = highlight_fill
    elif metric == '':
        # Empty separator rows
        cell_metric.border = None
        cell_count.border = None
        cell_percentage.border = None

# Adjust column widths
ws.column_dimensions['A'].width = 30
ws.column_dimensions['B'].width = 15
ws.column_dimensions['C'].width = 15

# Save workbook
wb.save(output_file)

print(f"✓ Successfully saved Batch Generation Statistics to '{output_file}'")
print(f"  Location: /Users/suppboat/Generative_CRAM_Fragment/{output_file}")
print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")
