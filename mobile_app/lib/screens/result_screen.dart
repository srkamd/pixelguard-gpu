import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import '../services/forensic_engine.dart';

class ResultScreen extends StatelessWidget {
  final ForensicResult result;

  const ResultScreen({super.key, required this.result});

  void _shareResult() {
    final text = "PixelGuard Forensic Inspection Result:\n"
        "Verdict: ${result.isAuthentic ? 'VERIFIED AUTHENTIC' : 'POTENTIAL FORGERY DETECTED'}\n"
        "Risk Score: ${result.riskScore}%\n"
        "Inference Speed: ${result.latencyMs} ms (100% On-Device)\n"
        "Protected by PixelGuard AI";
    Share.share(text);
  }

  @override
  Widget build(BuildContext context) {
    final isAuth = result.isAuthentic;
    final primaryColor = isAuth ? const Color(0xFF34D399) : const Color(0xFFF87171);
    final borderColor = isAuth ? const Color(0xFF065F46) : const Color(0xFF7F1D1D);
    final bgColor = isAuth ? const Color(0xFF064E3B).withOpacity(0.3) : const Color(0xFF450A0A).withOpacity(0.3);

    return Scaffold(
      appBar: AppBar(
        title: const Text("Inspection Report", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_outlined),
            onPressed: _shareResult,
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              // Verdict Card
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 24, horizontal: 16),
                decoration: BoxDecoration(
                  color: bgColor,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: borderColor, width: 1.5),
                ),
                child: Column(
                  children: [
                    Icon(
                      isAuth ? Icons.verified_user_rounded : Icons.warning_amber_rounded,
                      size: 54,
                      color: primaryColor,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      isAuth ? "AUTHENTIC DOCUMENT" : "FORGERY DETECTED",
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 0.5,
                        color: primaryColor,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      result.summary,
                      textAlign: TextAlign.center,
                      style: const TextStyle(fontSize: 13, color: Color(0xFFCBD5E1), height: 1.4),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),

              // Metrics Grid
              Row(
                children: [
                  Expanded(
                    child: _buildMetricTile("Risk Score", "${result.riskScore}%", primaryColor),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildMetricTile("Authenticity", "${result.authenticityProb.toStringAsFixed(1)}%", primaryColor),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildMetricTile("Scan Time", "${result.latencyMs} ms", const Color(0xFF3B82F6)),
                  ),
                ],
              ),
              const SizedBox(height: 16),

              // Document Preview Card
              Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: const Color(0xFF1E293B),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: const Color(0xFF334155)),
                ),
                clipBehavior: Clip.antiAlias,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(12),
                      child: Row(
                        children: const [
                          Icon(Icons.image_outlined, size: 16, color: Color(0xFF94A3B8)),
                          SizedBox(width: 8),
                          Text(
                            "Inspected Document Preview",
                            style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: Color(0xFF94A3B8)),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(
                      height: 260,
                      width: double.infinity,
                      child: Image.file(result.file, fit: BoxFit.contain),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // Google Policy Compliant Disclaimer
              const Text(
                "Disclaimer: PixelGuard provides automated algorithmic image noise and compression residual inspection for informational and digital hygiene purposes.",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 11, color: Color(0xFF64748B), height: 1.4),
              ),
              const SizedBox(height: 16),

              // Done Button
              SizedBox(
                width: double.infinity,
                child: OutlinedButton(
                  style: OutlinedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 14),
                    side: const BorderSide(color: Color(0xFF475569)),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  onPressed: () => Navigator.pop(context),
                  child: const Text("Scan Another Document", style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold)),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMetricTile(String label, String value, Color valueColor) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1E293B),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFF334155)),
      ),
      child: Column(
        children: [
          Text(
            label.toUpperCase(),
            style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: Color(0xFF94A3B8)),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: valueColor),
          ),
        ],
      ),
    );
  }
}
