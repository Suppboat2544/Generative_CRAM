#!/usr/bin/env python3
"""
Export tables from CRAM paper to Excel
Creates an Excel file with two sheets containing:
1. Property Conditioning Accuracy
2. Performance vs Baseline Generative Models
"""

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# Table 1: Property Conditioning Accuracy
property_conditioning_data = {
    'Target Property': [
        'COOH groups',
        'COOH groups',
        'Aliphatic %',
        'O/C ratio',
        'Rings'
    ],
    'Target Value': [
        '2',
        '4',
        '75%',
        '0.5',
        '1'
    ],
    'Mean Generated': [
        '2.12 ± 0.38',
        '3.89 ± 0.56',
        '74.1% ± 8.4%',
        '0.51 ± 0.09',
        '1.23 ± 0.58'
    ],
    'Hit Rate': [
        '84.2%',
        '76.8%',
        '68.3%',
        '72.6%',
        '71.4%'
    ]
}

# Table 2: Performance vs Baseline Generative Models
baseline_comparison_data = {
    'Model': [
        'CharRNN',
        'JT-VAE',
        'GraphVAE',
        'MolGAN',
        'Our Model'
    ],
    'Validity': [
        '72.4%',
        '89.6%',
        '85.2%',
        '91.3%',
        '97.2%'
    ],
    'CRAM Compliance': [
        '12.8%',
        '23.4%',
        '18.9%',
        '27.6%',
        '39.2%'
    ],
    'Novelty': [
        '81.2%',
        '88.5%',
        '85.7%',
        '90.1%',
        '92.3%'
    ],
    'Diversity': [
        '0.48',
        '0.62',
        '0.54',
        '0.68',
        '0.72'
    ]
}

# Create DataFrames
df_conditioning = pd.DataFrame(property_conditioning_data)
df_baseline = pd.DataFrame(baseline_comparison_data)

# Create Excel file with openpyxl for better formatting
wb = Workbook()

# Remove default sheet
if 'Sheet' in wb.sheetnames:
    wb.remove(wb['Sheet'])

# Define styles
header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
header_font = Font(bold=True, color='FFFFFF', size=11)
highlight_fill = PatternFill(start_color='FFF2CC', end_color='FFF2CC', fill_type='solid')
highlight_font = Font(bold=True)
border = Border(
    left=Side(style='thin'),
    right=Side(style='thin'),
    top=Side(style='thin'),
    bottom=Side(style='thin')
)

# Sheet 1: Property Conditioning Accuracy
ws1 = wb.create_sheet('Property Conditioning')
ws1.append(['Property Conditioning Accuracy'])
ws1.append([])  # Empty row

# Merge and style title
ws1.merge_cells('A1:D1')
ws1['A1'].font = Font(bold=True, size=14)
ws1['A1'].alignment = Alignment(horizontal='center')

# Add headers
for col_idx, header in enumerate(df_conditioning.columns, 1):
    cell = ws1.cell(row=3, column=col_idx, value=header)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center')
    cell.border = border

# Add data
for row_idx, row in enumerate(dataframe_to_rows(df_conditioning, index=False, header=False), 4):
    for col_idx, value in enumerate(row, 1):
        cell = ws1.cell(row=row_idx, column=col_idx, value=value)
        cell.border = border
        cell.alignment = Alignment(horizontal='center')

# Adjust column widths
ws1.column_dimensions['A'].width = 20
ws1.column_dimensions['B'].width = 15
ws1.column_dimensions['C'].width = 20
ws1.column_dimensions['D'].width = 12

# Sheet 2: Performance vs Baseline Models
ws2 = wb.create_sheet('Baseline Comparison')
ws2.append(['Performance vs Baseline Generative Models'])
ws2.append([])  # Empty row

# Merge and style title
ws2.merge_cells('A1:E1')
ws2['A1'].font = Font(bold=True, size=14)
ws2['A1'].alignment = Alignment(horizontal='center')

# Add headers
for col_idx, header in enumerate(df_baseline.columns, 1):
    cell = ws2.cell(row=3, column=col_idx, value=header)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center')
    cell.border = border

# Add data
for row_idx, row in enumerate(dataframe_to_rows(df_baseline, index=False, header=False), 4):
    for col_idx, value in enumerate(row, 1):
        cell = ws2.cell(row=row_idx, column=col_idx, value=value)
        cell.border = border
        cell.alignment = Alignment(horizontal='center')
        
        # Highlight "Our Model" row
        if row_idx == 8:  # Last row (Our Model)
            cell.fill = highlight_fill
            cell.font = highlight_font

# Adjust column widths
ws2.column_dimensions['A'].width = 15
ws2.column_dimensions['B'].width = 12
ws2.column_dimensions['C'].width = 18
ws2.column_dimensions['D'].width = 12
ws2.column_dimensions['E'].width = 12

# Save the Excel file
output_file = 'CRAM_Model_Tables.xlsx'
wb.save(output_file)

print(f"✅ Excel file created successfully: {output_file}")
print(f"\nSheets created:")
print(f"  1. Property Conditioning - {len(df_conditioning)} rows")
print(f"  2. Baseline Comparison - {len(df_baseline)} rows")
print(f"\nFile saved to: {output_file}")
