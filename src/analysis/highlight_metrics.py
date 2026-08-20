import re

with open('project_overview.md', 'r') as f:
    content = f.read()

# Fix the 432 typo
content = content.replace('432', '378')

# Process the table
lines = content.split('\n')
new_lines = []
in_table = False

for line in lines:
    if line.startswith('| COPPER |') or line.startswith('| CRUDE_OIL |') or line.startswith('| GOLD |') or line.startswith('| NATURAL_GAS |') or line.startswith('| SILVER |') or line.startswith('| WHEAT |'):
        parts = line.split('|')
        
        # Extracted values
        dir_diff_str = parts[6].strip()
        mae_diff_str = parts[9].strip()
        
        try:
            dir_diff = float(dir_diff_str.replace('%', '').replace('+', ''))
            
            # Highlight logic: If improvement > 15%, make the row bold or highlight the specific cell
            if dir_diff >= 15.0:
                parts[6] = f" **{dir_diff_str}** 🚀 "
                parts[1] = f" **{parts[1].strip()}** "
                parts[2] = f" **{parts[2].strip()}** "
                parts[3] = f" **{parts[3].strip()}** "
            elif dir_diff >= 5.0:
                parts[6] = f" **{dir_diff_str}** "
            
            mae_diff = float(mae_diff_str.replace('+', ''))
            if mae_diff >= 1.0:
                 parts[9] = f" **{mae_diff_str}** ⭐ "
                 
        except ValueError:
            pass
            
        new_lines.append('|'.join(parts))
    else:
        new_lines.append(line)

with open('project_overview.md', 'w') as f:
    f.write('\n'.join(new_lines))
