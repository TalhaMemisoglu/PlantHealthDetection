"""
Evaluate on test set with confusion matrix visualization.
Run: python3 run_test_eval.py
"""
from main import PlantHealthDetector, evaluate_dataset
import os

def run_evaluation():
    """Run evaluation on both training sample and test set with confusion matrices."""
    detector = PlantHealthDetector()
    base_dir = '/Users/talha/Desktop/Projects/DIP'
    
    # Paths
    healthy_dir = os.path.join(base_dir, 'healthy')
    unhealthy_dir = os.path.join(base_dir, 'unhealthy')
    healthy_test_dir = os.path.join(base_dir, 'healthy_test')
    unhealthy_test_dir = os.path.join(base_dir, 'unhealthy_test')
    
    print("="*60)
    print("PLANT HEALTH DETECTION - EVALUATION")
    print("="*60)
    
    # 1. Training Sample (30+30) - same size as test
    print("\n[1/2] TRAINING SAMPLE (30+30)")
    print("-"*40)
    train_results = evaluate_dataset(detector, healthy_dir, unhealthy_dir, 
                                     sample_size=30, 
                                     title="Training Sample (30+30)")
    
    # 2. Test Set (30+30) - unseen data
    print("\n[2/2] TEST SET - UNSEEN DATA (30+30)")
    print("-"*40)
    test_results = evaluate_dataset(detector, healthy_test_dir, unhealthy_test_dir,
                                    sample_size=None,
                                    title="Test Set - Unseen Data (30+30)")
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    
    train_acc = (train_results['healthy']['correct'] + train_results['unhealthy']['correct']) / \
                (train_results['healthy']['total'] + train_results['unhealthy']['total']) * 100
    test_acc = (test_results['healthy']['correct'] + test_results['unhealthy']['correct']) / \
               (test_results['healthy']['total'] + test_results['unhealthy']['total']) * 100
    
    print(f"\nTraining Sample Accuracy: {train_acc:.1f}%")
    print(f"Test Set Accuracy:        {test_acc:.1f}%")
    print(f"Difference:               {train_acc - test_acc:+.1f}%")
    print("="*60)
    
    return train_results, test_results

if __name__ == '__main__':
    run_evaluation()

