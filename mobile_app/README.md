# 📱 PixelGuard Flutter Mobile App

An ultra-lightweight, 100% on-device AI document integrity scanner for Android & iOS built with Flutter and PyTorch Lite.

---

## 🌟 Key Features
- **100% On-Device & Offline:** Powered by the embedded `pixelguard_engine_opt.ptl` (3.90 MB) model in `assets/models/`.
- **Zero Cloud Costs:** Runs directly on phone hardware (Snapdragon / MediaTek) with $0.00 server hosting bills.
- **Google Play 2026 Compliant:** Uses Android Photo Picker without requiring dangerous `MANAGE_EXTERNAL_STORAGE` permissions.
- **Privacy-First:** Guarantees zero sensitive receipt data leaves the user's smartphone.

---

## 🚀 How to Run and Build

### 1. Requirements
- Flutter SDK 3.10+
- Android Studio / Android SDK 34+

### 2. Install Packages
```bash
flutter pub get
```

### 3. Test on Phone / Emulator
Connect your Android phone via USB (with Developer options & USB debugging enabled):
```bash
flutter run
```

### 4. Build Production Release Bundle for Google Play
To generate the `.aab` file required by Google Play Console:
```bash
flutter build appbundle --release
```
The output file will be generated at:
`build/app/outputs/bundle/release/app-release.aab`

To generate an installable `.apk` for instant testing on your own phone:
```bash
flutter build apk --release
```

---

## 🔒 Google Play Console Listing Information

- **App Name:** `PixelGuard: Document Inspector`
- **Category:** `Tools` / `Productivity`
- **Target Audience:** Age 18+ (Productivity / Business)
- **Privacy Policy URL:** `https://github.com/srkamd/pixelguard-gpu/blob/main/privacy-policy.html`
- **Data Safety Declaration:**
  - *Data collected:* None (Select "No, this app does not collect or share user data").
  - *Data encrypted in transit:* Not applicable (No network transmission of user documents).
