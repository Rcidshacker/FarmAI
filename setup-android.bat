@echo off
setlocal enabledelayedexpansion

echo ============================================
echo FarmAI - Android Development Environment Setup
echo ============================================
echo.

REM Check Java
echo Checking Java JDK...
java -version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✓ Java JDK found
    java -version
    echo.
) else (
    echo.
    echo [ERROR] Java JDK not installed or not in PATH!
    echo.
    echo Please install Java JDK 17 from:
    echo https://www.oracle.com/java/technologies/downloads/#java17
    echo.
    pause
    exit /b 1
)

REM Check Android SDK
echo Checking Android SDK...
if defined ANDROID_HOME (
    echo ✓ ANDROID_HOME already set to: !ANDROID_HOME!
) else (
    if exist "%LOCALAPPDATA%\Android\Sdk" (
        echo ✓ Found Android SDK at default location
        setx ANDROID_HOME "%LOCALAPPDATA%\Android\Sdk"
        set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
        echo ✓ Set ANDROID_HOME to: !ANDROID_HOME!
    ) else (
        echo.
        echo [ERROR] Android SDK not found!
        echo.
        echo Please install Android Studio from:
        echo https://developer.android.com/studio
        echo.
        pause
        exit /b 1
    )
)
echo.

REM Create local.properties for Android project
echo Creating local.properties for Android build...
if not exist "frontend\android" (
    echo [ERROR] Android project folder not found!
    pause
    exit /b 1
)

cd frontend\android

if not defined ANDROID_HOME (
    if exist "%LOCALAPPDATA%\Android\Sdk" (
        set "ANDROID_HOME=%LOCALAPPDATA%\Android\Sdk"
    ) else (
        echo [ERROR] Cannot determine Android SDK path
        cd ..\..
        pause
        exit /b 1
    )
)

REM Quote the path to handle spaces in directory names
(
    echo sdk.dir=!ANDROID_HOME!
) > local.properties

echo ✓ Created local.properties with SDK path
echo   sdk.dir=!ANDROID_HOME!
echo.

cd ..\..

echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo Your system is now ready to build APKs!
echo.
echo To build the FarmAI APK, run:
echo   .\build-apk.bat
echo.
pause
