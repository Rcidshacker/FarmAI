@echo off
setlocal enabledelayedexpansion
echo ============================================
echo FarmAI Mobile App - APK Build Script
echo ============================================
echo.

REM Check Java installation
echo [0/5] Checking Java installation...
java -version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Java JDK not found!
    echo Please install Java JDK 17 from: https://www.oracle.com/java/technologies/downloads/
    echo After installation, set JAVA_HOME environment variable.
    echo.
    pause
    exit /b 1
)
echo ✓ Java found
echo.

REM Check Android SDK
echo [0.5/5] Checking Android SDK installation...
if defined ANDROID_HOME (
    echo ✓ ANDROID_HOME is set
) else if exist "%LOCALAPPDATA%\Android\Sdk" (
    echo ✓ Found Android SDK in default location
    set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
) else (
    echo ERROR: Android SDK not found!
    echo Please install Android Studio: https://developer.android.com/studio
    echo.
    pause
    exit /b 1
)
echo.

echo [1/5] Building React Web App...
cd frontend
call npm run build
if errorlevel 1 (
    echo ERROR: Web build failed!
    pause
    exit /b 1
)
echo ✓ Web app built successfully
echo.

echo [2/5] Syncing to Android...
call npx cap sync android
if errorlevel 1 (
    echo ERROR: Sync failed!
    pause
    exit /b 1
)
echo ✓ Synced successfully
echo.

echo [3/5] Creating local.properties...
cd android
if not defined ANDROID_HOME (
    if exist "%LOCALAPPDATA%\Android\Sdk" (
        set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
    )
)
REM Write SDK path to local.properties with escaped backslashes for Gradle
setlocal enabledelayedexpansion
set "SDK_PATH=!ANDROID_HOME:\=\\!"
(
    echo sdk.dir=!SDK_PATH!
) > local.properties
endlocal
echo ✓ local.properties created
echo.

echo [4/5] Building Debug APK...
echo This may take 5-10 minutes on first build...
echo.
call .\gradlew.bat assembleDebug
if errorlevel 1 (
    echo.
    echo ERROR: Android build failed!
    echo.
    echo Solutions:
    echo 1. Run: .\setup-android.bat
    echo 2. Check that ANDROID_HOME points to valid SDK
    echo.
    pause
    exit /b 1
)
echo.
echo ✓ APK built successfully
echo.

echo [5/5] Locating APK...
echo.
if exist "app\build\outputs\apk\debug\app-debug.apk" (
    echo ============================================
    echo BUILD SUCCESSFUL!
    echo ============================================
    echo.
    echo Debug APK location:
    echo !cd!\app\build\outputs\apk\debug\app-debug.apk
    echo.
    echo To install on device:
    echo 1. Enable USB Debugging on your Android device
    echo 2. Connect device via USB
    echo 3. Run: adb install app\build\outputs\apk\debug\app-debug.apk
    echo.
    echo OR copy the APK to your device and install manually
    echo.
) else (
    echo ERROR: APK not found at expected location!
)

cd ..\..
pause
