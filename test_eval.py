#!/usr/bin/env python3
"""
Quick test script for FedSense evaluation functionality.
"""

import sys
import os
sys.path.insert(0, '/Users/rohansriram/FedSense')

def test_evaluation():
    print("🧪 Testing FedSense Evaluation Functions")
    print("=" * 45)
    
    try:
        # Test imports
        from fedsense.eval import compute_classification_metrics, find_optimal_threshold
        import numpy as np
        print("✅ Imports successful")
        
        # Generate test data
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.binomial(1, 0.2, n_samples)  # 20% anomalies
        y_scores = np.random.beta(2, 8, n_samples)
        y_scores[y_true == 1] += 0.4  # Boost anomaly scores
        
        print(f"✅ Generated {n_samples} test samples ({y_true.mean():.1%} anomalies)")
        
        # Test metrics computation
        metrics = compute_classification_metrics(y_true, y_scores, threshold=0.5)
        print(f"✅ Metrics computed:")
        print(f"   AUROC: {metrics['auroc']:.4f}")
        print(f"   F1: {metrics['f1']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        
        # Test threshold optimization
        opt_threshold, opt_f1 = find_optimal_threshold(y_true, y_scores, 'f1')
        print(f"✅ Optimal threshold: {opt_threshold:.3f} (F1={opt_f1:.4f})")
        
        print("\n🎉 All evaluation tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation()
