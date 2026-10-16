import os
import sys
import glob

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

from app import analyze_document

pos_dir = os.path.join('SAMPLE DATASET FOR IMAGE DETECTION', 'positive orignal')
neg_dir = os.path.join('SAMPLE DATASET FOR IMAGE DETECTION', 'negative fake')

pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if os.path.isfile(os.path.join(pos_dir, f))]
neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if os.path.isfile(os.path.join(neg_dir, f))]

print("=" * 80)
print("PIXELGUARD-GPU NEURAL DATASET BENCHMARK")
print("=" * 80)

print("\n--- EVALUATING POSITIVE (ORIGINAL) SAMPLES ---")
pos_correct = 0
pos_results = []
for p in pos_files:
    fname = os.path.basename(p)
    try:
        with open(p, 'rb') as f:
            b = f.read()
        res = analyze_document(b)
        status = 'PASS (Authentic)' if res['is_authentic'] else 'FAIL (Flagged Fake!)'
        if res['is_authentic']:
            pos_correct += 1
        pos_results.append((fname, res['risk_score'], res['neural_authenticity_prob'], len(res['anomaly_boxes']), res['detected_software'], status))
        print(f"{fname[:30]:30} | Risk: {res['risk_score']:3}% | Auth: {res['neural_authenticity_prob']:5}% | Boxes: {len(res['anomaly_boxes']):2} | SW: {res['detected_software']} | {status}")
    except Exception as e:
        print(f"{fname[:30]:30} | ERROR: {e}")

print(f"\nPositive Accuracy: {pos_correct}/{len(pos_files)} ({pos_correct/len(pos_files)*100:.1f}%)")

print("\n--- EVALUATING NEGATIVE (FAKE) SAMPLES ---")
neg_correct = 0
neg_results = []
for p in neg_files:
    fname = os.path.basename(p)
    try:
        with open(p, 'rb') as f:
            b = f.read()
        res = analyze_document(b)
        status = 'PASS (Detected Fake)' if not res['is_authentic'] else 'FAIL (Missed Fake!)'
        if not res['is_authentic']:
            neg_correct += 1
        neg_results.append((fname, res['risk_score'], res['neural_authenticity_prob'], len(res['anomaly_boxes']), res['detected_software'], status))
        print(f"{fname[:30]:30} | Risk: {res['risk_score']:3}% | Auth: {res['neural_authenticity_prob']:5}% | Boxes: {len(res['anomaly_boxes']):2} | SW: {res['detected_software']} | {status}")
    except Exception as e:
        print(f"{fname[:30]:30} | ERROR: {e}")

print(f"\nNegative Accuracy: {neg_correct}/{len(neg_files)} ({neg_correct/len(neg_files)*100:.1f}%)")
total_acc = (pos_correct + neg_correct) / (len(pos_files) + len(neg_files)) * 100
print(f"\nOverall Accuracy: {pos_correct + neg_correct}/{len(pos_files) + len(neg_files)} ({total_acc:.1f}%)")
print("=" * 80)
