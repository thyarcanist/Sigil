import os
import json

def format_result(result_dict):
    """Formats a single test result dictionary for display."""
    if result_dict is None:
        return "Test Skipped or Errored (Result is None)"
    
    parts = []
    for key, value in result_dict.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.6f}")
        elif value is None:
            parts.append(f"{key}: N/A")
        else:
            parts.append(f"{key}: {value}")
    return ", ".join(parts)

def generate_text_summary(results: dict, output_path: str = None):
    """Generates a text summary of the analysis results.

    Args:
        results: A dictionary containing the results from various analyzers.
                 Expected keys: 'label', 'filepath', 'stats', 'fft_path', 'wavelet_path', 'visual_path'
        output_path: Optional path to save the summary file. If None, prints to console.
    """
    label = results.get('label', 'Unknown Label')
    filepath = results.get('filepath', 'Unknown File')
    
    summary_lines = []
    summary_lines.append(f"=== SIGIL Analysis Summary for: {label} ===")
    summary_lines.append(f"Source File: {filepath}")
    summary_lines.append("-" * (len(summary_lines[0])))
    
    # Statistical Tests
    summary_lines.append("\n[Statistical Tests]")
    stats_results = results.get('stats', {})
    if stats_results:
        freq_res = stats_results.get('frequency_monobit')
        chisq_res = stats_results.get('chi_square_byte')
        runs_res = stats_results.get('runs')
        
        summary_lines.append(f"  Frequency (Monobit): {format_result(freq_res)}")
        summary_lines.append(f"  Chi-Square (Bytes):  {format_result(chisq_res)}")
        summary_lines.append(f"  Runs Test:           {format_result(runs_res)}")
    else:
        summary_lines.append("  No statistical test results available.")
        
    # Transform/Visual Analysis Outputs
    summary_lines.append("\n[Visual & Transform Outputs]")
    summary_lines.append(f"  Bit Visualization: {results.get('visual_path', 'Not Generated')}")
    summary_lines.append(f"  FFT Spectrum Plot: {results.get('fft_path', 'Not Generated')}")
    summary_lines.append(f"  Wavelet Decomp Plot: {results.get('wavelet_path', 'Not Generated')}")
    
    summary_lines.append("\n=== End of Summary ===")
    
    summary_text = "\n".join(summary_lines)
    
    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(summary_text)
            print(f"Analysis summary saved to: {output_path}")
        except Exception as e:
            print(f"Error saving summary to {output_path}: {e}")
            print("\n--- Summary --- ")
            print(summary_text)
            print("---------------")
    else:
        print("\n--- Summary --- ")
        print(summary_text)
        print("---------------")

# TODO: Add function for generating JSON report
# TODO: Add function for generating HTML report (potentially using Jinja2)
