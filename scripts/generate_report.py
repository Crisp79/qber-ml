#!/usr/bin/env python3
"""
Generate PDF report from saved experiment results.

Usage:
    uv run python scripts/generate_report.py                    # Generate from saved results
    uv run python scripts/generate_report.py --execute-notebook # Re-execute notebook first
    uv run python scripts/generate_report.py --no-notebook      # Skip notebook execution
    uv run python scripts/generate_report.py --output PATH      # Custom output path
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
REPORT_DIR = PROJECT_ROOT / "report"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "qad_experiments.ipynb"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"


def check_requirements():
    """Check for required files and LaTeX installation."""
    errors = []
    warnings = []

    # Check required files
    required_files = [
        REPORT_DIR / "main.tex",
        REPORT_DIR / "references.bib",
        RESULTS_DIR / "evaluation_results.json",
    ]
    for f in required_files:
        if not f.exists():
            errors.append(f"Missing required file: {f}")

    # Check LaTeX installation (pdflatex)
    pdflatex_path = r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
    if not Path(pdflatex_path).exists():
        errors.append(
            "pdflatex not found. Install MiKTeX (https://miktex.org/download) "
            "and ensure it's in PATH or at the default location."
        )
    else:
        print(f"Found pdflatex: {pdflatex_path}")

    # Check figures exist
    figure_dir = REPORT_DIR / "figures"
    if not figure_dir.exists():
        warnings.append(f"Figure directory not found: {figure_dir}")
    else:
        expected_figs = [
            "class_distribution.png",
            "feature_correlation.png",
            "model_comparison.png",
            "confusion_matrices.png",
            "per_class_f1.png",
            "roc_curves.png",
            "shap_importance.png",
            "ablation.png",
        ]
        for fig in expected_figs:
            if not (figure_dir / fig).exists():
                warnings.append(f"Missing figure: {figure_dir / fig}")

    return errors, warnings, pdflatex_path


def execute_notebook():
    """Execute the Jupyter notebook to refresh outputs."""
    print("Executing notebook...")
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--execute", "--to", "notebook",
        "--output", str(NOTEBOOK_PATH),
        "--ExecutePreprocessor.timeout=600",
        "--ExecutePreprocessor.kernel_name=python3",
        str(NOTEBOOK_PATH)
    ]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"Notebook execution failed:\n{result.stderr}")
        return False
    print("Notebook executed successfully")
    return True


def copy_figures_to_report():
    """Copy generated figures to report/figures directory."""
    src = RESULTS_DIR
    dst = REPORT_DIR / "figures"
    dst.mkdir(parents=True, exist_ok=True)

    figure_files = list(src.glob("*.png"))
    copied = 0
    for fig in figure_files:
        shutil.copy2(fig, dst / fig.name)
        copied += 1
    print(f"Copied {copied} figures to {dst}")
    return copied > 0


def generate_results_table():
    """Generate LaTeX table from evaluation results."""
    with open(RESULTS_DIR / "evaluation_results.json") as f:
        data = json.load(f)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Multiclass (8-class) Classification Results on Test Set (n=2,000)}")
    lines.append(r"\label{tab:multiclass_results}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Macro-Prec} & \textbf{Macro-Rec} & \textbf{ROC-AUC (OvR)} \\")
    lines.append(r"\midrule")

    for model_name, model_data in data["models"].items():
        eval_data = model_data.get("evaluation", {})
        acc = eval_data.get("accuracy", 0)
        f1 = eval_data.get("macro_f1", 0)
        prec = eval_data.get("macro_precision", 0)
        rec = eval_data.get("macro_recall", 0)
        auc = eval_data.get("roc_auc_ovr", 0)
        display_name = model_name.replace("_", " ").title()
        lines.append(f"{display_name} & {acc:.4f} & {f1:.4f} & {prec:.4f} & {rec:.4f} & {auc:.4f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_path = REPORT_DIR / "generated" / "results_table.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines))
    print(f"Generated results table: {table_path}")
    return str(table_path)


def generate_ablation_table():
    """Generate LaTeX ablation table from evaluation results."""
    with open(RESULTS_DIR / "evaluation_results.json") as f:
        data = json.load(f)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Feature-Group Ablation: Accuracy and Macro-F1 Drop from Baseline}")
    lines.append(r"\label{tab:ablation}")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Group} & \textbf{N} & \multicolumn{2}{c}{\textbf{XGBoost}} & \multicolumn{2}{c}{\textbf{Random Forest}} & \textbf{SVM-RBF} \\")
    lines.append(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}")
    lines.append(r" & & $\Delta$Acc & $\Delta$F1 & $\Delta$Acc & $\Delta$F1 & $\Delta$Acc \\")
    lines.append(r"\midrule")

    # Get group order from first model
    first_model = list(data["models"].values())[0]
    ablation = first_model.get("ablation", {})
    groups = list(ablation.get("ablation_results", {}).keys())

    for group in groups:
        n_feat = ablation["ablation_results"][group]["n_features_removed"]
        row = [group.replace("_", r"\_"), str(n_feat)]

        for model_key in ["xgboost", "random_forest", "svm_rbf"]:
            model_data = data["models"].get(model_key, {})
            model_ablation = model_data.get("ablation", {})
            group_result = model_ablation.get("ablation_results", {}).get(group, {})
            drop_acc = group_result.get("drop_accuracy", 0)
            drop_f1 = group_result.get("drop_f1", 0)

            if model_key == "svm_rbf":
                row.append(f"{drop_acc*100:.1f}\\%")
            else:
                row.append(f"{drop_acc*100:.1f}\\% & {drop_f1*100:.1f}\\%")

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    table_path = REPORT_DIR / "generated" / "ablation_table.tex"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("\n".join(lines))
    print(f"Generated ablation table: {table_path}")
    return str(table_path)


def compile_latex(latexmk_path):
    """Compile LaTeX to PDF using pdflatex directly (more reliable on Windows)."""
    print("Compiling LaTeX...")
    pdflatex_path = r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
    
    # Run pdflatex 3 times for cross-references, bibliography, etc.
    for i in range(3):
        print(f"  pdflatex pass {i+1}/3...")
        cmd = [
            pdflatex_path,
            "-interaction=nonstopmode",
            "-output-directory=generated",
            "main.tex"
        ]
        result = subprocess.run(cmd, cwd=REPORT_DIR, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"pdflatex pass {i+1} failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    
    # Check PDF was created
    pdf_path = REPORT_DIR / "generated" / "main.pdf"
    if pdf_path.exists():
        print(f"PDF generated: {pdf_path}")
        return True
    else:
        print("PDF not found after compilation")
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate PDF report")
    parser.add_argument("--execute-notebook", action="store_true",
                        help="Re-execute notebook before generating report")
    parser.add_argument("--no-notebook", action="store_true",
                        help="Skip notebook execution (use saved outputs)")
    parser.add_argument("--output", type=str,
                        help="Custom output PDF path")
    args = parser.parse_args()

    print("=" * 60)
    print("QBER-ML Report Generator")
    print("=" * 60)

    # Check requirements
    errors, warnings, latexmk_path = check_requirements()

    for w in warnings:
        print(f"WARNING: {w}")

    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # Execute notebook if requested
    if args.execute_notebook and not args.no_notebook:
        if not execute_notebook():
            print("Notebook execution failed, continuing with saved outputs...")

    # Copy figures
    if not copy_figures_to_report():
        print("WARNING: No figures copied")

    # Generate dynamic tables
    generate_results_table()
    generate_ablation_table()

    # Compile LaTeX
    if not compile_latex(latexmk_path):
        sys.exit(1)

    # Copy to custom output if specified
    pdf_path = REPORT_DIR / "generated" / "main.pdf"
    if args.output:
        out_path = Path(args.output).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, out_path)
        print(f"Copied PDF to: {out_path}")
        pdf_path = out_path

    print("=" * 60)
    print(f"Report generated successfully: {pdf_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()