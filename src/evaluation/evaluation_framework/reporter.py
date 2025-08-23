"""
Reporting engine for the evaluation framework.
Generates reports in multiple formats with visualization support.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import pandas as pd
from io import StringIO
import html

from .models import (
    EvaluationRun, 
    EvaluationResult, 
    ComparisonResult,
    TestCase
)
from .metrics_calculator import AggregatedMetrics
from .config import ReportingConfig


class ReportFormat(Enum):
    """Available report formats."""
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    HTML = "html"
    LATEX = "latex"


class BaseReporter:
    """
    Base class for report generation.
    Defines the interface for different report formats.
    """
    
    def __init__(self, config: ReportingConfig):
        """
        Initialize reporter with configuration.
        
        Args:
            config: Reporting configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_filename(
        self,
        base_name: str,
        format: ReportFormat,
        timestamp: bool = True
    ) -> Path:
        """
        Generate output filename.
        
        Args:
            base_name: Base name for the file
            format: Report format
            timestamp: Whether to include timestamp
            
        Returns:
            Full path to output file
        """
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_{ts}.{format.value}"
        else:
            filename = f"{base_name}.{format.value}"
        
        return self.output_dir / filename


class JSONReporter(BaseReporter):
    """Reporter for JSON format."""
    
    def export_run(self, run: EvaluationRun, filename: Optional[Path] = None) -> Path:
        """
        Export evaluation run to JSON.
        
        Args:
            run: Evaluation run to export
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename(f"evaluation_{run.name}", ReportFormat.JSON)
        
        data = run.to_dict()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filename
    
    def export_comparison(
        self,
        comparison: ComparisonResult,
        filename: Optional[Path] = None
    ) -> Path:
        """
        Export comparison results to JSON.
        
        Args:
            comparison: Comparison results
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename("comparison", ReportFormat.JSON)
        
        data = comparison.to_dict()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filename


class CSVReporter(BaseReporter):
    """Reporter for CSV format."""
    
    def export_run(self, run: EvaluationRun, filename: Optional[Path] = None) -> Path:
        """
        Export evaluation run to CSV.
        
        Args:
            run: Evaluation run to export
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename(f"evaluation_{run.name}", ReportFormat.CSV)
        
        # Create DataFrame from results
        rows = []
        for result in run.results:
            row = {
                "test_id": result.test_case.id,
                "query": result.test_case.query[:100],  # Truncate long queries
                "num_expected": len(result.test_case.expected_tools),
                "num_recommended": len(result.recommended_tools),
                "execution_time_ms": result.execution_time_ms,
                "status": result.status.value
            }
            
            # Add metrics
            for metric_name, value in result.metrics.items():
                if not isinstance(value, (list, dict, set)):
                    row[metric_name] = value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        return filename
    
    def export_comparison(
        self,
        comparison: ComparisonResult,
        filename: Optional[Path] = None
    ) -> Path:
        """
        Export comparison results to CSV.
        
        Args:
            comparison: Comparison results
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename("comparison", ReportFormat.CSV)
        
        # Create DataFrame from comparison metrics
        rows = []
        for run_id, metrics in comparison.comparison_metrics.items():
            run_name = next((r.name for r in comparison.runs if r.id == run_id), run_id)
            row = {"run_id": run_id, "run_name": run_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
        
        return filename


class MarkdownReporter(BaseReporter):
    """Reporter for Markdown format."""
    
    def export_run(self, run: EvaluationRun, filename: Optional[Path] = None) -> Path:
        """
        Export evaluation run to Markdown.
        
        Args:
            run: Evaluation run to export
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename(f"evaluation_{run.name}", ReportFormat.MARKDOWN)
        
        # Generate Markdown content
        lines = [
            f"# Evaluation Report: {run.name}",
            "",
            f"**Run ID**: {run.id}",
            f"**Start Time**: {run.start_time.isoformat()}",
            f"**End Time**: {run.end_time.isoformat() if run.end_time else 'N/A'}",
            f"**Duration**: {run.duration_seconds:.2f}s" if run.duration_seconds else "",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(run.config, indent=2),
            "```",
            "",
            "## Summary Statistics",
            "",
            f"- **Total Test Cases**: {len(run.results)}",
            f"- **Successful**: {run.num_successful}",
            f"- **Failed**: {run.num_failed}",
            f"- **Success Rate**: {run.success_rate:.2%}",
            "",
            "## Aggregated Metrics",
            ""
        ]
        
        # Add aggregated metrics table
        aggregated = run.aggregate_metrics()
        if aggregated:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for metric, value in sorted(aggregated.items()):
                if isinstance(value, float):
                    lines.append(f"| {metric} | {value:.4f} |")
                else:
                    lines.append(f"| {metric} | {value} |")
        
        lines.extend([
            "",
            "## Detailed Results",
            "",
            "| Test ID | Query | Precision | Recall | F1 | MRR | P@1 | Time (ms) | Status |",
            "|---------|-------|-----------|--------|-------|-----|-----|-----------|---------|"
        ])
        
        # Add individual results
        for result in run.results[:100]:  # Limit to first 100 for readability
            query_short = result.test_case.query[:30] + "..." if len(result.test_case.query) > 30 else result.test_case.query
            query_short = query_short.replace("|", "\\|")  # Escape pipes
            
            lines.append(
                f"| {result.test_case.id} | {query_short} | "
                f"{result.metrics.get('precision', 0):.3f} | "
                f"{result.metrics.get('recall', 0):.3f} | "
                f"{result.metrics.get('f1_score', 0):.3f} | "
                f"{result.metrics.get('mrr', 0):.3f} | "
                f"{result.metrics.get('p@1', 0):.3f} | "
                f"{result.execution_time_ms:.1f} | "
                f"{result.status.value} |"
            )
        
        if len(run.results) > 100:
            lines.append(f"\n*Note: Showing first 100 of {len(run.results)} results*")
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        
        return filename
    
    def export_comparison(
        self,
        comparison: ComparisonResult,
        filename: Optional[Path] = None
    ) -> Path:
        """
        Export comparison results to Markdown.
        
        Args:
            comparison: Comparison results
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename("comparison", ReportFormat.MARKDOWN)
        
        lines = [
            "# Evaluation Comparison Report",
            "",
            f"**Generated**: {datetime.now().isoformat()}",
            "",
            "## Runs Compared",
            ""
        ]
        
        # List runs
        for run in comparison.runs:
            lines.append(f"- **{run.name}** (ID: {run.id})")
        
        lines.extend([
            "",
            "## Metrics Comparison",
            "",
            "| Metric | " + " | ".join(run.name for run in comparison.runs) + " | Best |",
            "|--------|" + "|".join("-------" for _ in comparison.runs) + "|------|"
        ])
        
        # Get all metric names
        all_metrics = set()
        for metrics in comparison.comparison_metrics.values():
            all_metrics.update(metrics.keys())
        
        # Add metric rows
        for metric in sorted(all_metrics):
            row = [f"| {metric}"]
            values = []
            
            for run in comparison.runs:
                value = comparison.comparison_metrics.get(run.id, {}).get(metric, "N/A")
                if isinstance(value, float):
                    row.append(f" {value:.4f}")
                    values.append((run.id, value))
                else:
                    row.append(f" {value}")
            
            # Determine best
            if values:
                best_id = max(values, key=lambda x: x[1])[0] if "loss" not in metric.lower() else min(values, key=lambda x: x[1])[0]
                best_name = next((r.name for r in comparison.runs if r.id == best_id), "")
                row.append(f" **{best_name}** |")
            else:
                row.append(" - |")
            
            lines.append("".join(row))
        
        # Add best performers section
        lines.extend([
            "",
            "## Best Performers by Metric",
            ""
        ])
        
        for metric, run_id in comparison.best_run_by_metric.items():
            run_name = next((r.name for r in comparison.runs if r.id == run_id), run_id)
            value = comparison.comparison_metrics.get(run_id, {}).get(metric, "N/A")
            if isinstance(value, float):
                lines.append(f"- **{metric}**: {run_name} ({value:.4f})")
            else:
                lines.append(f"- **{metric}**: {run_name} ({value})")
        
        # Add statistical tests if available
        if comparison.statistical_tests:
            lines.extend([
                "",
                "## Statistical Significance",
                ""
            ])
            
            for test_name, results in comparison.statistical_tests.items():
                lines.append(f"### {test_name}")
                lines.append("")
                lines.append("```")
                lines.append(str(results))
                lines.append("```")
                lines.append("")
        
        # Write to file
        with open(filename, 'w') as f:
            f.write("\n".join(lines))
        
        return filename


class HTMLReporter(BaseReporter):
    """Reporter for HTML format with visualizations."""
    
    def export_run(self, run: EvaluationRun, filename: Optional[Path] = None) -> Path:
        """
        Export evaluation run to HTML with interactive visualizations.
        
        Args:
            run: Evaluation run to export
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename(f"evaluation_{run.name}", ReportFormat.HTML)
        
        # Generate HTML content
        html_content = self._generate_html_report(run)
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        return filename
    
    def _generate_html_report(self, run: EvaluationRun) -> str:
        """Generate HTML report with charts."""
        aggregated = run.aggregate_metrics()
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Report: {title}</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .chart {{
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .good {{ color: #4CAF50; }}
        .warning {{ color: #FF9800; }}
        .bad {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Report: {title}</h1>
        
        <div class="summary">
            <div class="stat-card">
                <div class="stat-label">Total Tests</div>
                <div class="stat-value">{total_tests}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4CAF50 0%, #8BC34A 100%);">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value">{success_rate:.1%}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #FF6B6B 0%, #FFE66D 100%);">
                <div class="stat-label">Mean F1 Score</div>
                <div class="stat-value">{mean_f1:.3f}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);">
                <div class="stat-label">Mean MRR</div>
                <div class="stat-value">{mean_mrr:.3f}</div>
            </div>
        </div>
        
        <h2>Metrics Distribution</h2>
        <div id="metrics-dist" class="chart"></div>
        
        <h2>Performance Over Time</h2>
        <div id="performance-time" class="chart"></div>
        
        <h2>Top Performing Test Cases</h2>
        <table>
            <thead>
                <tr>
                    <th>Test ID</th>
                    <th>Query</th>
                    <th>F1 Score</th>
                    <th>MRR</th>
                    <th>P@1</th>
                    <th>Time (ms)</th>
                </tr>
            </thead>
            <tbody>
                {top_results_rows}
            </tbody>
        </table>
        
        <h2>Configuration</h2>
        <pre style="background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;">
{config}
        </pre>
    </div>
    
    <script>
        // Metrics Distribution Chart
        var metricsData = [
            {{
                x: {f1_scores},
                type: 'histogram',
                name: 'F1 Score',
                marker: {{ color: '#4CAF50' }}
            }},
            {{
                x: {precisions},
                type: 'histogram',
                name: 'Precision',
                marker: {{ color: '#2196F3' }}
            }},
            {{
                x: {recalls},
                type: 'histogram',
                name: 'Recall',
                marker: {{ color: '#FF9800' }}
            }}
        ];
        
        var metricsLayout = {{
            title: 'Metrics Distribution',
            xaxis: {{ title: 'Value' }},
            yaxis: {{ title: 'Count' }},
            barmode: 'overlay',
            showlegend: true
        }};
        
        Plotly.newPlot('metrics-dist', metricsData, metricsLayout);
        
        // Performance Over Time Chart
        var performanceData = [
            {{
                x: {test_indices},
                y: {f1_scores},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'F1 Score',
                line: {{ color: '#4CAF50' }}
            }},
            {{
                x: {test_indices},
                y: {mrr_scores},
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MRR',
                line: {{ color: '#FF6B6B' }}
            }}
        ];
        
        var performanceLayout = {{
            title: 'Performance Across Test Cases',
            xaxis: {{ title: 'Test Index' }},
            yaxis: {{ title: 'Score' }},
            showlegend: true
        }};
        
        Plotly.newPlot('performance-time', performanceData, performanceLayout);
    </script>
</body>
</html>
        """
        
        # Extract metrics for charts
        successful_results = [r for r in run.results if r.is_successful]
        
        f1_scores = [r.metrics.get('f1_score', 0) for r in successful_results]
        precisions = [r.metrics.get('precision', 0) for r in successful_results]
        recalls = [r.metrics.get('recall', 0) for r in successful_results]
        mrr_scores = [r.metrics.get('mrr', 0) for r in successful_results]
        
        # Get top results
        top_results = sorted(successful_results, key=lambda r: r.metrics.get('f1_score', 0), reverse=True)[:10]
        top_results_rows = []
        for r in top_results:
            query_short = html.escape(r.test_case.query[:50] + "..." if len(r.test_case.query) > 50 else r.test_case.query)
            f1_class = "good" if r.metrics.get('f1_score', 0) > 0.8 else "warning" if r.metrics.get('f1_score', 0) > 0.5 else "bad"
            
            top_results_rows.append(f"""
                <tr>
                    <td>{r.test_case.id}</td>
                    <td>{query_short}</td>
                    <td class="{f1_class}">{r.metrics.get('f1_score', 0):.3f}</td>
                    <td>{r.metrics.get('mrr', 0):.3f}</td>
                    <td>{r.metrics.get('p@1', 0):.3f}</td>
                    <td>{r.execution_time_ms:.1f}</td>
                </tr>
            """)
        
        return html_template.format(
            title=run.name,
            total_tests=len(run.results),
            success_rate=run.success_rate,
            mean_f1=aggregated.get('mean_f1_score', 0),
            mean_mrr=aggregated.get('mean_mrr', 0),
            f1_scores=f1_scores,
            precisions=precisions,
            recalls=recalls,
            mrr_scores=mrr_scores,
            test_indices=list(range(len(successful_results))),
            top_results_rows="\n".join(top_results_rows),
            config=json.dumps(run.config, indent=2)
        )
    
    def export_comparison(
        self,
        comparison: ComparisonResult,
        filename: Optional[Path] = None
    ) -> Path:
        """
        Export comparison results to HTML.
        
        Args:
            comparison: Comparison results
            filename: Optional output filename
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = self.generate_filename("comparison", ReportFormat.HTML)
        
        html_content = self._generate_comparison_html(comparison)
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        return filename
    
    def _generate_comparison_html(self, comparison: ComparisonResult) -> str:
        """Generate HTML comparison report."""
        
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Evaluation Comparison Report</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .chart {{
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .best {{
            background: #e8f5e9;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evaluation Comparison Report</h1>
        
        <h2>Runs Compared</h2>
        <ul>
            {runs_list}
        </ul>
        
        <h2>Metrics Comparison</h2>
        <div id="comparison-chart" class="chart"></div>
        
        <h2>Detailed Comparison Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    {run_headers}
                </tr>
            </thead>
            <tbody>
                {metric_rows}
            </tbody>
        </table>
        
        <h2>Best Performers</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Best Run</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {best_rows}
            </tbody>
        </table>
    </div>
    
    <script>
        {chart_script}
    </script>
</body>
</html>
        """
        
        # Generate runs list
        runs_list = "\n".join([f"<li><strong>{run.name}</strong> (ID: {run.id})</li>" for run in comparison.runs])
        
        # Generate headers
        run_headers = "\n".join([f"<th>{run.name}</th>" for run in comparison.runs])
        
        # Generate metric rows
        all_metrics = set()
        for metrics in comparison.comparison_metrics.values():
            all_metrics.update(metrics.keys())
        
        metric_rows = []
        for metric in sorted(all_metrics):
            row = [f"<td><strong>{metric}</strong></td>"]
            best_value = None
            best_run_id = comparison.best_run_by_metric.get(metric)
            
            for run in comparison.runs:
                value = comparison.comparison_metrics.get(run.id, {}).get(metric, "N/A")
                is_best = run.id == best_run_id
                
                if isinstance(value, float):
                    cell_class = "best" if is_best else ""
                    row.append(f'<td class="{cell_class}">{value:.4f}</td>')
                else:
                    row.append(f"<td>{value}</td>")
            
            metric_rows.append("<tr>" + "".join(row) + "</tr>")
        
        # Generate best performers rows
        best_rows = []
        for metric, run_id in comparison.best_run_by_metric.items():
            run_name = next((r.name for r in comparison.runs if r.id == run_id), run_id)
            value = comparison.comparison_metrics.get(run_id, {}).get(metric, "N/A")
            
            if isinstance(value, float):
                best_rows.append(f"""
                    <tr>
                        <td>{metric}</td>
                        <td><strong>{run_name}</strong></td>
                        <td>{value:.4f}</td>
                    </tr>
                """)
        
        # Generate chart script
        chart_data = []
        for run in comparison.runs:
            metrics = comparison.comparison_metrics.get(run.id, {})
            
            # Select key metrics for visualization
            key_metrics = ['mean_f1_score', 'mean_precision', 'mean_recall', 'mean_mrr', 'mean_p@1']
            values = [metrics.get(m, 0) for m in key_metrics]
            
            chart_data.append(f"""
                {{
                    x: {key_metrics},
                    y: {values},
                    type: 'bar',
                    name: '{run.name}'
                }}
            """)
        
        chart_script = f"""
            var data = [{','.join(chart_data)}];
            
            var layout = {{
                title: 'Key Metrics Comparison',
                xaxis: {{ title: 'Metric' }},
                yaxis: {{ title: 'Value' }},
                barmode: 'group'
            }};
            
            Plotly.newPlot('comparison-chart', data, layout);
        """
        
        return html_template.format(
            runs_list=runs_list,
            run_headers=run_headers,
            metric_rows="\n".join(metric_rows),
            best_rows="\n".join(best_rows),
            chart_script=chart_script
        )


class EvaluationReporter:
    """
    Main reporting engine that coordinates different report formats.
    """
    
    def __init__(self, config: ReportingConfig):
        """
        Initialize the evaluation reporter.
        
        Args:
            config: Reporting configuration
        """
        self.config = config
        
        # Initialize format-specific reporters
        self.reporters = {
            ReportFormat.JSON: JSONReporter(config),
            ReportFormat.CSV: CSVReporter(config),
            ReportFormat.MARKDOWN: MarkdownReporter(config),
            ReportFormat.HTML: HTMLReporter(config)
        }
    
    def export_run(
        self,
        run: EvaluationRun,
        formats: Optional[List[ReportFormat]] = None
    ) -> Dict[ReportFormat, Path]:
        """
        Export evaluation run in multiple formats.
        
        Args:
            run: Evaluation run to export
            formats: List of formats to export (None = use config)
            
        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = [
                ReportFormat(fmt) for fmt in self.config.export_formats
            ]
        
        exported_files = {}
        
        for format in formats:
            if format in self.reporters:
                reporter = self.reporters[format]
                filepath = reporter.export_run(run)
                exported_files[format] = filepath
                print(f"Exported {format.value} report to: {filepath}")
            else:
                print(f"Warning: No reporter available for format {format.value}")
        
        return exported_files
    
    def export_comparison(
        self,
        comparison: ComparisonResult,
        formats: Optional[List[ReportFormat]] = None
    ) -> Dict[ReportFormat, Path]:
        """
        Export comparison results in multiple formats.
        
        Args:
            comparison: Comparison results to export
            formats: List of formats to export (None = use config)
            
        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = [
                ReportFormat(fmt) for fmt in self.config.export_formats
            ]
        
        exported_files = {}
        
        for format in formats:
            if format in self.reporters:
                reporter = self.reporters[format]
                filepath = reporter.export_comparison(comparison)
                exported_files[format] = filepath
                print(f"Exported {format.value} comparison to: {filepath}")
            else:
                print(f"Warning: No reporter available for format {format.value}")
        
        return exported_files
    
    def generate_summary(self, run: EvaluationRun) -> str:
        """
        Generate a text summary of an evaluation run.
        
        Args:
            run: Evaluation run
            
        Returns:
            Text summary
        """
        aggregated = run.aggregate_metrics()
        
        summary_lines = [
            f"Evaluation Summary: {run.name}",
            "=" * 50,
            f"Run ID: {run.id}",
            f"Duration: {run.duration_seconds:.2f}s" if run.duration_seconds else "Duration: N/A",
            "",
            "Test Statistics:",
            f"  Total: {len(run.results)}",
            f"  Successful: {run.num_successful}",
            f"  Failed: {run.num_failed}",
            f"  Success Rate: {run.success_rate:.2%}",
            "",
            "Key Metrics (mean):"
        ]
        
        # Add key metrics
        key_metrics = [
            ('Precision', 'mean_precision'),
            ('Recall', 'mean_recall'),
            ('F1 Score', 'mean_f1_score'),
            ('MRR', 'mean_mrr'),
            ('P@1', 'mean_p@1'),
            ('NDCG@10', 'mean_ndcg@10')
        ]
        
        for label, metric_key in key_metrics:
            value = aggregated.get(metric_key, 0)
            if isinstance(value, float):
                summary_lines.append(f"  {label}: {value:.4f}")
        
        return "\n".join(summary_lines)
    
    def generate_comparison_summary(self, comparison: ComparisonResult) -> str:
        """
        Generate a text summary of comparison results.
        
        Args:
            comparison: Comparison results
            
        Returns:
            Text summary
        """
        return comparison.generate_summary()