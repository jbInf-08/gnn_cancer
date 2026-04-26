import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
from PIL import Image
import json
import time

RESULTS_DIR = 'results'
DATA_PROCESSED_DIR = 'data/processed'
SUMMARY_CSV = os.path.join(RESULTS_DIR, 'summary_table.csv')
REPORT_PDF = os.path.join(RESULTS_DIR, 'report.pdf')

# 1. Find all relevant plot files
def find_plots():
    patterns = [
        '*_learning_curves.png',
        'tsne_visualization.png',
        'attention_map.png',
        'feature_importance.png',
        'learning_curves.png',
        'gene_network.png'
    ]
    plot_files = []
    for pat in patterns:
        # Check in results directory
        plot_files.extend(glob.glob(os.path.join(RESULTS_DIR, pat)))
        # Check in data/processed directory
        plot_files.extend(glob.glob(os.path.join(DATA_PROCESSED_DIR, pat)))
    return plot_files

# 2. Load metrics from JSON files (prioritize data/processed)
def load_metrics():
    # First try to load from data/processed/metrics.json (most recent)
    metrics_file = os.path.join(DATA_PROCESSED_DIR, 'metrics.json')
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            # Extract test metrics
            test_metrics = metrics_data.get('test_metrics', {})
            dataset_stats = metrics_data.get('dataset_stats', {})
            
            # Create summary row
            summary_rows = [{
                'Model': 'EnhancedGAT',
                'Dataset_Size': f"{dataset_stats.get('total_nodes', 'N/A'):,}" if dataset_stats.get('total_nodes') else 'N/A',
                'Test_Size': f"{test_metrics.get('test_set_size', 'N/A'):,}" if test_metrics.get('test_set_size') else 'N/A',
                'Accuracy': f"{test_metrics.get('accuracy', 'N/A'):.4f}" if test_metrics.get('accuracy') is not None else 'N/A',
                'F1_Score': f"{test_metrics.get('f1_score', 'N/A'):.4f}" if test_metrics.get('f1_score') is not None else 'N/A',
                'Precision': f"{test_metrics.get('precision', 'N/A'):.4f}" if test_metrics.get('precision') is not None else 'N/A',
                'Recall': f"{test_metrics.get('recall', 'N/A'):.4f}" if test_metrics.get('recall') is not None else 'N/A',
                'ROC_AUC': f"{test_metrics.get('auc_roc', 'N/A'):.4f}" if test_metrics.get('auc_roc') is not None else 'N/A',
                'Status': 'Training Complete' if test_metrics.get('test_set_size', 0) > 100 else 'Training in Progress'
            }]
            
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(SUMMARY_CSV, index=False)
            return summary_df
            
        except Exception as e:
            print(f"Error loading metrics from {metrics_file}: {e}")
    
    # Fallback: try to find all metrics CSVs or JSONs in results/
    metrics_files = glob.glob(os.path.join(RESULTS_DIR, '*.csv')) + glob.glob(os.path.join(RESULTS_DIR, '*.json'))
    summary_rows = []
    for f in metrics_files:
        if 'summary_table' in f:
            continue
        if f.endswith('.csv'):
            df = pd.read_csv(f)
            if {'F1', 'Accuracy', 'Precision', 'Recall', 'Loss'}.issubset(df.columns):
                for _, row in df.iterrows():
                    summary_rows.append(row)
        elif f.endswith('.json'):
            df = pd.read_json(f)
            if {'F1', 'Accuracy', 'Precision', 'Recall', 'Loss'}.issubset(df.columns):
                for _, row in df.iterrows():
                    summary_rows.append(row)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV, index=False)
        return summary_df
    else:
        # If no metrics found, create a status table
        summary_df = pd.DataFrame({
            'Model': ['EnhancedGAT'],
            'Status': ['Training in Progress'],
            'Dataset_Size': ['967,189 nodes'],
            'Test_Size': ['145,079 nodes (expected)'],
            'Accuracy': ['Pending'],
            'F1_Score': ['Pending'],
            'Precision': ['Pending'],
            'Recall': ['Pending'],
            'ROC_AUC': ['Pending']
        })
        summary_df.to_csv(SUMMARY_CSV, index=False)
        return summary_df

# 3. Generate PDF report with all plots and summary table
def generate_pdf_report(summary_df, plot_files):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'GNN Cancer Mutation Analysis Report', ln=True, align='C')
    pdf.ln(10)
    
    # Add timestamp
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on: {time.strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
    pdf.ln(5)
    
    # Add summary table
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Model Performance Summary:', ln=True)
    pdf.ln(5)
    
    if not summary_df.empty:
        # Calculate column widths
        col_widths = []
        for col in summary_df.columns:
            max_width = len(str(col))
            for _, row in summary_df.iterrows():
                max_width = max(max_width, len(str(row[col])))
            col_widths.append(min(max_width * 2, 40))  # Cap at 40
        
        # Add table headers
        pdf.set_font('Arial', 'B', 10)
        for i, col in enumerate(summary_df.columns):
            pdf.cell(col_widths[i], 8, str(col), border=1)
        pdf.ln(8)
        
        # Add table rows
        pdf.set_font('Arial', '', 9)
        for _, row in summary_df.iterrows():
            for i, col in enumerate(summary_df.columns):
                pdf.cell(col_widths[i], 8, str(row[col]), border=1)
            pdf.ln(8)
    
    pdf.ln(10)
    
    # Add plots with better quality
    if plot_files:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Visualizations:', ln=True)
        pdf.ln(5)
        
        for plot in plot_files:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 12)
            plot_name = os.path.basename(plot).replace('.png', '').replace('_', ' ').title()
            pdf.cell(0, 10, plot_name, ln=True)
            pdf.ln(5)
            
            try:
                # Open and process image with better quality
                img = Image.open(plot)
                w, h = img.size
                
                # Calculate better scaling - maintain aspect ratio, max width 500, max height 300
                max_w, max_h = 500, 300
                scale = min(max_w / w, max_h / h, 1)
                img_w, img_h = int(w * scale), int(h * scale)
                
                # Save temporary image with better quality
                temp_file = f'temp_plot_{os.path.basename(plot)}'
                img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)  # Better resampling
                img.save(temp_file, 'PNG', quality=95)
                
                # Add image to PDF
                pdf.image(temp_file, x=15, y=pdf.get_y(), w=img_w, h=img_h)
                pdf.ln(img_h + 10)
                
                # Clean up
                os.remove(temp_file)
                
            except Exception as e:
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 10, f'Error loading image: {str(e)}', ln=True)
    else:
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 10, 'No visualization plots found.', ln=True)
        pdf.cell(0, 10, 'Plots will be generated when training completes.', ln=True)
    
    pdf.output(REPORT_PDF)

if __name__ == '__main__':
    print("Generating GNN Cancer Mutation Analysis Report...")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Find plots
    plot_files = find_plots()
    print(f"Found {len(plot_files)} plot files: {[os.path.basename(f) for f in plot_files]}")
    
    # Load metrics
    summary_df = load_metrics()
    print(f"Loaded metrics for {len(summary_df)} model(s)")
    
    # Generate PDF report
    generate_pdf_report(summary_df, plot_files)
    print(f"Summary table saved to: {SUMMARY_CSV}")
    print(f"PDF report saved to: {REPORT_PDF}")
    
    # Check if training is still running
    import subprocess
    try:
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True, shell=True)
        if 'python.exe' in result.stdout:
            print("\n⚠️  NOTE: Python training process is still running.")
            print("   The report contains current available data.")
            print("   Re-run this script after training completes for final results.")
        else:
            print("\n✅ Training appears to be complete.")
    except:
        print("\nℹ️  Could not check training status.") 