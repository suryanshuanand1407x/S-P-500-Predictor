#!/usr/bin/env python3
"""
Streamlined Inference Runner for Sensex & Gold Price Predictions

This script provides a clean interface for running production inference
with comprehensive output formatting and result saving.

Usage:
    python run_inference.py                    # Run inference for all assets
    python run_inference.py --asset SENSEX     # Run for specific asset only
    python run_inference.py --save-json        # Save results to JSON file
    python run_inference.py --detailed         # Show detailed statistics
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from production_inference import ProductionInferenceEngine
import warnings
warnings.filterwarnings('ignore')


class InferenceRunner:
    """Streamlined inference runner with formatted output."""

    def __init__(self, config_path: str = 'configs/default.yaml'):
        """Initialize inference runner."""
        self.config_path = config_path
        self.engine = None

    def initialize(self) -> bool:
        """Initialize the inference engine."""
        try:
            print("🔧 Initializing production inference engine...")
            self.engine = ProductionInferenceEngine(
                self.config_path,
                validate_artifacts=True
            )
            print("✅ Engine initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize engine: {e}")
            return False

    def run_inference(self, assets: Optional[list] = None,
                     require_validation: bool = False) -> Dict[str, Any]:
        """
        Run inference for specified assets.

        Args:
            assets: List of assets to predict (None for all)
            require_validation: Whether to require backtest validation

        Returns:
            Dictionary with prediction results
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized. Call initialize() first.")

        print("\n🔮 Running production inference...\n")
        results = self.engine.predict(
            assets=assets,
            require_backtest_validation=require_validation
        )

        return results

    def format_prediction_output(self, results: Dict[str, Any],
                                detailed: bool = False) -> str:
        """Format prediction results for display."""
        lines = []
        lines.append("═" * 80)
        lines.append("📊 SENSEX & GOLD PRICE PREDICTIONS")
        lines.append("═" * 80)

        # Timestamp
        timestamp = datetime.fromisoformat(results['timestamp'])
        lines.append(f"⏰ Prediction Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"🔍 Backtest Validated: {'Yes' if results['backtest_validated'] else 'No'}")

        # Monitoring summary
        monitoring = results['monitoring']
        lines.append(f"\n📈 Monitoring Status:")
        lines.append(f"   • Drift Alerts: {monitoring['drift_alerts_count']}")
        lines.append(f"   • Guardrail Violations: {monitoring['guardrail_violations_count']}")
        lines.append(f"   • Total Predictions: {monitoring['total_predictions_made']}")

        # Predictions for each asset
        for asset, pred in results['predictions'].items():
            if 'error' in pred:
                lines.append(f"\n{'─' * 80}")
                lines.append(f"❌ {asset}: PREDICTION FAILED")
                lines.append(f"   Error: {pred['error']}")
                continue

            lines.append(f"\n{'─' * 80}")
            lines.append(f"📌 {asset}")
            lines.append(f"{'─' * 80}")

            # Current price
            current = pred['current_price']
            predicted = pred['final_prediction']
            change_pct = pred['price_change_pct']
            direction = pred['direction']

            # Direction emoji
            dir_emoji = "🔴" if direction == "DOWN" else "🟢" if direction == "UP" else "⚪"

            lines.append(f"\n💰 Current Price:     ₹{current:,.2f}")
            lines.append(f"🎯 Predicted Price:   ₹{predicted:,.2f}")
            lines.append(f"{dir_emoji} Price Change:      {change_pct:+.2f}%")
            lines.append(f"📍 Direction:         {direction}")

            # Ensemble statistics
            if detailed:
                ensemble = pred['ensemble_stats']
                lines.append(f"\n📊 Ensemble Statistics:")
                lines.append(f"   • Mean Prediction:   ₹{ensemble['prediction']:,.2f}")
                lines.append(f"   • Std Deviation:     ₹{ensemble['std']:,.2f}")
                lines.append(f"   • Min Prediction:    ₹{ensemble['min']:,.2f}")
                lines.append(f"   • Max Prediction:    ₹{ensemble['max']:,.2f}")
                lines.append(f"   • Median:            ₹{ensemble['median']:,.2f}")
                lines.append(f"   • 95% CI:            [₹{ensemble['confidence_95_lower']:,.2f}, "
                           f"₹{ensemble['confidence_95_upper']:,.2f}]")
                lines.append(f"   • Models Used:       {ensemble['model_count']}")

            # Production flags
            flags = pred['production_flags']
            lines.append(f"\n🛡️ Production Flags:")
            lines.append(f"   • Schema Validated:   {'✅' if flags['schema_validated'] else '❌'}")
            lines.append(f"   • Drift Detected:     {'⚠️ Yes' if flags['drift_detected'] else '✅ No'}")
            lines.append(f"   • Guardrails Applied: {'⚠️ Yes' if flags['guardrails_applied'] else '✅ No'}")
            lines.append(f"   • Sanity Checks:      {'✅ Pass' if flags['sanity_checks_passed'] else '❌ Fail'}")

            # Drift information
            drift = pred['drift_info']
            drift_level = drift.get('drift_level', 'unknown')
            drift_emoji = "🔴" if drift_level == "critical" else "🟡" if drift_level == "warning" else "🟢"
            lines.append(f"\n📊 Distribution Analysis:")
            lines.append(f"   • Drift Level:        {drift_emoji} {drift_level.upper()}")
            lines.append(f"   • Drift Score:        {drift.get('overall_drift_score', 0):.2f}")

            # Guardrail information
            if flags['guardrails_applied']:
                guardrail = pred['guardrail_info']
                lines.append(f"\n⚠️ Guardrail Details:")
                lines.append(f"   • Raw Prediction:     ₹{guardrail['raw_prediction']:,.2f}")
                lines.append(f"   • Raw Change:         {guardrail['raw_change_pct']:+.2f}%")
                lines.append(f"   • Adjusted To:        ±{guardrail['guardrail_threshold']:.1f}% limit")

        lines.append(f"\n{'═' * 80}")

        # Warnings and recommendations
        has_warnings = False
        warnings_list = []

        for asset, pred in results['predictions'].items():
            if 'error' not in pred:
                if pred['production_flags']['drift_detected']:
                    warnings_list.append(f"• {asset}: Critical distribution drift detected - prediction reliability may be reduced")
                    has_warnings = True
                if pred['production_flags']['guardrails_applied']:
                    warnings_list.append(f"• {asset}: Extreme prediction capped by safety guardrails")
                    has_warnings = True

        if has_warnings:
            lines.append("\n⚠️ WARNINGS:")
            lines.extend(warnings_list)
            lines.append("\n💡 Recommendation: Review model performance and consider retraining with recent data.")

        lines.append("═" * 80)

        return "\n".join(lines)

    def save_results(self, results: Dict[str, Any], output_dir: str = 'predictions') -> str:
        """Save prediction results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"prediction_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        return filepath

    def save_summary(self, results: Dict[str, Any], output_dir: str = 'predictions') -> str:
        """Save a simplified summary CSV for tracking."""
        os.makedirs(output_dir, exist_ok=True)

        summary_file = os.path.join(output_dir, 'prediction_history.csv')

        # Create or append to CSV
        import csv

        file_exists = os.path.exists(summary_file)

        with open(summary_file, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write header if new file
            if not file_exists:
                writer.writerow([
                    'Timestamp', 'Asset', 'Current_Price', 'Predicted_Price',
                    'Change_%', 'Direction', 'Drift_Level', 'Guardrails_Applied'
                ])

            # Write predictions
            for asset, pred in results['predictions'].items():
                if 'error' not in pred:
                    writer.writerow([
                        results['timestamp'],
                        asset,
                        f"{pred['current_price']:.2f}",
                        f"{pred['final_prediction']:.2f}",
                        f"{pred['price_change_pct']:+.2f}",
                        pred['direction'],
                        pred['drift_info'].get('drift_level', 'unknown'),
                        'Yes' if pred['production_flags']['guardrails_applied'] else 'No'
                    ])

        return summary_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run production inference for Sensex & Gold price predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py                     # Run inference for all assets
  python run_inference.py --asset SENSEX      # Predict SENSEX only
  python run_inference.py --asset GOLD        # Predict GOLD only
  python run_inference.py --detailed          # Show detailed statistics
  python run_inference.py --save-json         # Save full JSON results
  python run_inference.py --no-validation     # Skip backtest validation check
        """
    )

    parser.add_argument('--asset', type=str, choices=['SENSEX', 'GOLD'],
                       help='Specific asset to predict (default: all)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Configuration file path')
    parser.add_argument('--detailed', action='store_true',
                       help='Show detailed prediction statistics')
    parser.add_argument('--save-json', action='store_true',
                       help='Save full results to JSON file')
    parser.add_argument('--no-validation', action='store_true',
                       help='Skip backtest validation requirement')
    parser.add_argument('--output-dir', type=str, default='predictions',
                       help='Output directory for saved predictions')

    args = parser.parse_args()

    # Initialize runner
    runner = InferenceRunner(args.config)

    if not runner.initialize():
        sys.exit(1)

    # Determine assets to predict
    assets = [args.asset] if args.asset else None

    try:
        # Run inference
        results = runner.run_inference(
            assets=assets,
            require_validation=not args.no_validation
        )

        # Display formatted output
        output = runner.format_prediction_output(results, detailed=args.detailed)
        print(output)

        # Save results if requested
        if args.save_json:
            json_path = runner.save_results(results, args.output_dir)
            print(f"\n💾 Full results saved to: {json_path}")

        # Always save summary to CSV for tracking
        csv_path = runner.save_summary(results, args.output_dir)
        print(f"📝 Summary appended to: {csv_path}")

        print("\n✅ Inference completed successfully!")

    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
