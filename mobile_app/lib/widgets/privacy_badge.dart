import 'package:flutter/material.dart';

class PrivacyBadge extends StatelessWidget {
  const PrivacyBadge({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF064E3B).withOpacity(0.4), // Dark emerald
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: const Color(0xFF059669).withOpacity(0.6)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: const [
          Icon(Icons.shield_outlined, color: Color(0xFF34D399), size: 16),
          SizedBox(width: 8),
          Text(
            "100% On-Device & Private — No Cloud Uploads",
            style: TextStyle(
              color: Color(0xFF34D399),
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }
}
