"""
Test to verify macro-averaging fixes the metrics gap
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.threshold_optimizer import ThresholdOptimizer

def test_macro_vs_micro_averaging():
    """
    Demonstrate the difference between macro and micro averaging
    and why macro-averaging matches actual system behavior
    """
    optimizer = ThresholdOptimizer()
    
    # Simulate two queries with very different performance
    
    # Query 1: Good performance
    # Expected: tool_a, tool_b
    # tool_a scores 0.6 (relevant, will be retrieved)
    # tool_b scores 0.5 (relevant, will be retrieved)  
    # tool_c scores 0.3 (irrelevant, won't be retrieved)
    optimizer.add_score(0.6, True, "tool_a", query_id=1)
    optimizer.add_score(0.5, True, "tool_b", query_id=1)
    optimizer.add_score(0.3, False, "tool_c", query_id=1)
    
    # Query 2: Poor performance
    # Expected: tool_d, tool_e
    # tool_d scores 0.35 (relevant, won't be retrieved!)
    # tool_e scores 0.30 (relevant, won't be retrieved!)
    # tool_f scores 0.45 (irrelevant, will be retrieved!)
    optimizer.add_score(0.35, True, "tool_d", query_id=2)
    optimizer.add_score(0.30, True, "tool_e", query_id=2)
    optimizer.add_score(0.45, False, "tool_f", query_id=2)
    
    # Set threshold at 0.4
    threshold = 0.4
    
    # Calculate with micro-averaging (old bug)
    micro_metrics = optimizer.calculate_metrics_at_threshold(threshold, use_macro_averaging=False)
    
    # Calculate with macro-averaging (fixed)
    macro_metrics = optimizer.calculate_metrics_at_threshold(threshold, use_macro_averaging=True)
    
    print("=" * 60)
    print("Macro vs Micro Averaging Demonstration")
    print("=" * 60)
    print(f"\nThreshold: {threshold}")
    
    print("\n--- Query 1 Performance ---")
    print("Expected: tool_a (0.6), tool_b (0.5)")
    print("Retrieved: tool_a (0.6), tool_b (0.5)")
    print("Precision: 2/2 = 1.0, Recall: 2/2 = 1.0")
    
    print("\n--- Query 2 Performance ---")
    print("Expected: tool_d (0.35), tool_e (0.30)")  
    print("Retrieved: tool_f (0.45)")
    print("Precision: 0/1 = 0.0, Recall: 0/2 = 0.0")
    
    print("\n" + "=" * 60)
    print("MICRO-AVERAGING (Bug - Wrong for IR):")
    print(f"  Precision: {micro_metrics['precision']:.3f}")
    print(f"  Recall: {micro_metrics['recall']:.3f}")
    print(f"  F1: {micro_metrics['f1']:.3f}")
    print("  Note: Pools all tools together, misleading!")
    
    print("\nMACRO-AVERAGING (Fixed - IR Standard):")
    print(f"  Precision: {macro_metrics['precision']:.3f}")
    print(f"  Recall: {macro_metrics['recall']:.3f}")
    print(f"  F1: {macro_metrics['f1']:.3f}")
    print("  Note: Average of per-query metrics, accurate!")
    
    print("\n" + "=" * 60)
    print("ACTUAL SYSTEM BEHAVIOR:")
    print("  Query 1: Precision=1.0, Recall=1.0")
    print("  Query 2: Precision=0.0, Recall=0.0")
    print("  Average: Precision=0.5, Recall=0.5")
    print("\nMacro-averaging matches actual system!")
    print("Micro-averaging creates misleading metrics gap!")
    
    # Verify the difference
    assert abs(macro_metrics['precision'] - 0.5) < 0.01, "Macro precision should be 0.5"
    assert abs(macro_metrics['recall'] - 0.5) < 0.01, "Macro recall should be 0.5"
    
    # The key insight: micro-averaging gives different metrics than macro
    # In this case, micro gives higher precision (0.667 vs 0.5)
    assert abs(micro_metrics['precision'] - 0.667) < 0.01, "Micro precision is misleadingly high"
    assert abs(micro_metrics['recall'] - 0.5) < 0.01, "Micro recall happens to match in this example"
    
    # The important point: they give DIFFERENT results
    assert abs(micro_metrics['precision'] - macro_metrics['precision']) > 0.1, "Metrics should differ significantly"
    
    print("\n✓ Test passed: Macro-averaging correctly matches system behavior")
    print(f"✓ Micro gives misleading precision: {micro_metrics['precision']:.3f} vs actual {macro_metrics['precision']:.3f}")

if __name__ == "__main__":
    test_macro_vs_micro_averaging()