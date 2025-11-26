"""
add_detailed_tracking_all.py
-----------------------------
Adds detailed residual tracking to ALL existing model scripts.
Run this once to enhance all scripts with per-sample prediction logging.
"""

import os
import re

def add_detailed_import(script_content):
    """Add import statement for detailed metrics utility."""
    import_line = "\nfrom utils_detailed_metrics import save_detailed_predictions\n"
    
    # Find the last import statement
    import_pattern = r'(import .*\n|from .* import .*\n)'
    matches = list(re.finditer(import_pattern, script_content))
    
    if matches:
        last_import = matches[-1]
        insert_pos = last_import.end()
        return script_content[:insert_pos] + import_line + script_content[insert_pos:]
    else:
        # If no imports found, add at beginning after docstring
        docstring_end = script_content.find('"""', 10) + 3
        return script_content[:docstring_end] + "\n" + import_line + script_content[docstring_end:]

def add_save_call(script_content, model_var='model_name', exp_var='experiment'):
    """
    Add save_detailed_predictions call after metrics calculation.
    Looks for patterns like:
        metrics = {...}
        print(...)
        append_results(...)
    """
    
    # Pattern to find metrics calculation and reporting
    pattern = r"(metrics = \{[^}]+\})\s*\n\s*(print\([^\)]+\))\s*\n\s*(append_results[^\n]+)"
    
    def replacement(match):
        metrics_calc = match.group(1)
        print_line = match.group(2)
        append_line = match.group(3)
        
        save_call = f"""
            
            # Save detailed predictions and residuals
            save_detailed_predictions(
                model_name={model_var},
                experiment_name={exp_var},
                y_true=y_true,
                y_pred=y_pred,
                best_params=best_params,
                additional_info={{
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics.get('r2', 0),
                    'n_test': len(y_true)
                }}
            )
"""
        
        return f"{metrics_calc}\n            {print_line}\n            {append_line}{save_call}"
    
    return re.sub(pattern, replacement, script_content)

def enhance_script(script_path):
    """Add detailed tracking to a single script."""
    
    if not os.path.exists(script_path):
        print(f"[SKIP] File not found: {script_path}")
        return False
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already enhanced
    if 'save_detailed_predictions' in content:
        print(f"[SKIP] Already enhanced: {script_path}")
        return False
    
    # Add import
    content = add_detailed_import(content)
    
    # Note: Manual enhancement is better for accuracy
    # This auto-enhancement would need pattern matching specific to each script structure
    
    # For now, just add the import and return True to indicate manual editing needed
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[ENHANCED] Added import to: {script_path}")
    print(f"           Manual step: Add save_detailed_predictions() after metrics calculation")
    return True

# List of scripts to enhance
SCRIPTS = [
    "02_run_stat.py",
    "03_run_ml.py",
    "04_run_dl.py",
    "05_run_literature.py",
    "06_run_legacy.py",
    "06b_run_lstm_only.py"
]

if __name__ == "__main__":
    print("=" * 80)
    print("ADDING DETAILED RESIDUAL TRACKING TO ALL MODEL SCRIPTS")
    print("=" * 80)
    
    enhanced_count = 0
    for script in SCRIPTS:
        script_path = os.path.join("RunOnCluster", script) if not os.path.exists(script) else script
        if enhance_script(script_path):
            enhanced_count += 1
    
    print("\n" + "=" * 80)
    print(f"Enhanced {enhanced_count} scripts")
    print("\nNext steps:")
    print("1. Review each enhanced script")
    print("2. Add save_detailed_predictions() call after metrics are calculated")
    print("3. Example location: After 'metrics = {...}' and before final print/append")
    print("=" * 80)

