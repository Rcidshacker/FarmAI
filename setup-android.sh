#!/bin/bash

echo "============================================"
echo "FarmAI - Android Development Environment Setup"
echo "============================================"
echo ""

# Check Java
echo "Checking Java JDK..."
if ! command -v java &> /dev/null; then
    echo ""
    echo "[ERROR] Java JDK not installed!"
    echo ""
    echo "Please install Java JDK 17 from:"
    echo "https://www.oracle.com/java/technologies/downloads/#java17"
    echo ""
    echo "Installation steps:"
    echo "1. Download JDK 17 for your OS"
    echo "2. Run the installer"
    echo "3. Add Java to PATH if needed"
    echo "4. Run this script again after installation"
    echo ""
    exit 1
fi
echo "✓ Java JDK found"
java -version
echo ""

# Check Android SDK
echo "Checking Android SDK..."
if [ -z "$ANDROID_HOME" ]; then
    # Try default locations
    if [ -d "$HOME/Library/Android/sdk" ]; then
        export ANDROID_HOME="$HOME/Library/Android/sdk"
        echo "✓ Found Android SDK at default location (macOS)"
    elif [ -d "$HOME/Android/Sdk" ]; then
        export ANDROID_HOME="$HOME/Android/Sdk"
        echo "✓ Found Android SDK at default location (Linux)"
    else
        echo ""
        echo "[ERROR] Android SDK not found!"
        echo ""
        echo "Please install Android Studio from:"
        echo "https://developer.android.com/studio"
        echo ""
        echo "After installation, set ANDROID_HOME:"
        echo "export ANDROID_HOME=\$HOME/Library/Android/sdk  # macOS"
        echo "export ANDROID_HOME=\$HOME/Android/Sdk          # Linux"
        echo ""
        exit 1
    fi
else
    echo "✓ ANDROID_HOME is set to: $ANDROID_HOME"
fi

if [ -d "$ANDROID_HOME/platforms" ]; then
    echo "✓ SDK platforms found"
else
    echo "[WARNING] SDK platforms not found. You may need to install them in Android Studio."
fi
echo ""

# Create local.properties for Android project
echo "Creating local.properties for Android build..."
cd frontend/android
echo "sdk.dir=$ANDROID_HOME" > local.properties
echo "✓ Created local.properties"
echo "  sdk.dir=$ANDROID_HOME"
cd ../..
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Your system is now ready to build APKs!"
echo ""
echo "To build the FarmAI APK, run:"
echo "  ./build-apk.sh"
echo ""
echo "Or manually:"
echo "  cd frontend"
echo "  npm run build"
echo "  npx cap sync android"
echo "  cd android"
echo "  ./gradlew assembleDebug"
echo ""
