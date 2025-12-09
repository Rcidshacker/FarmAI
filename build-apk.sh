#!/bin/bash

echo "============================================"
echo "FarmAI Mobile App - APK Build Script"
echo "============================================"
echo ""

echo "[1/4] Building React Web App..."
cd frontend
npm run build
if [ $? -ne 0 ]; then
    echo "ERROR: Web build failed!"
    exit 1
fi
echo "✓ Web app built successfully"
echo ""

echo "[2/4] Syncing to Android..."
npx cap sync android
if [ $? -ne 0 ]; then
    echo "ERROR: Sync failed!"
    exit 1
fi
echo "✓ Synced successfully"
echo ""

echo "[3/4] Building Debug APK..."
cd android
./gradlew assembleDebug
if [ $? -ne 0 ]; then
    echo "ERROR: Android build failed!"
    echo ""
    echo "Make sure you have:"
    echo "- Android Studio installed"
    echo "- ANDROID_HOME environment variable set"
    echo "- Java JDK 17+ installed"
    exit 1
fi
echo "✓ APK built successfully"
echo ""

echo "[4/4] Locating APK..."
echo ""
echo "============================================"
echo "BUILD SUCCESSFUL!"
echo "============================================"
echo ""
echo "Debug APK location:"
echo "$(pwd)/app/build/outputs/apk/debug/app-debug.apk"
echo ""
echo "To install on device:"
echo "1. Enable USB Debugging on your Android device"
echo "2. Connect device via USB"
echo "3. Run: adb install app/build/outputs/apk/debug/app-debug.apk"
echo ""
echo "OR copy the APK to your device and install manually"
echo ""

cd ../..
