#!/bin/bash

# -----------------------------------------------------------------------------
# Currently (July 2025) the "usual way" of building LabRecorder on Apple Silicon 
# - as described in https://github.com/labstreaminglayer/App-LabRecorder -
# (using Homebrew and CMake) does not work due to Qt5 plugin path issues.
# This script provides a workaround to build LabRecorder successfully on M1/M2/M3 Macs
# while ensuring it is signed and SIP-compliant.

# 🧾 README — Build LabRecorder on Apple Silicon macOS (M1/M2/M3)
#
# This script builds a fully working, SIP-compliant, signed version of LabRecorder
# on Apple Silicon Macs using Homebrew packages and the LabRecorder source code.
#
# 🔧 What it does:
# - Installs required dependencies (cmake, qt@5, boost, lsl)
# - Clones the LabRecorder GitHub repo
# - Configures build for macOS/ARM64 via CMake
# - Builds and installs LabRecorder locally
# - Fixes Qt plugin paths to avoid runtime errors
# - Applies ad-hoc code signing to satisfy macOS Gatekeeper & SIP
# - Launches the resulting LabRecorder.app
#
# 🧱 Prerequisites:
# 1. ✅ Full Xcode.app installed (not just Command Line Tools)
#    - Download from App Store
#    - Then run:
#        sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
#        xcodebuild -runFirstLaunch
#
# 2. ✅ Homebrew installed: https://brew.sh
#
# ▶️ Usage:
#    bash build_labrecorder.sh
#
# 📁 Output:
#    ~/Applications/App-LabRecorder/install/LabRecorder/LabRecorder.app
#    (You can open this with Finder or run `open` in Terminal.)
#
# -----------------------------------------------------------------------------


set -euo pipefail
echo "🛠️ Starting LabRecorder build..."

# === 0. Prerequisites ===
echo "📦 Checking and installing dependencies..."

brew install cmake qt@5 boost labstreaminglayer/tap/lsl

# Ensure full Xcode is active
if ! xcode-select -p | grep -q Xcode.app; then
  echo "⚠️ Full Xcode required. Switching to full Xcode..."
  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
  xcodebuild -runFirstLaunch
fi

# === 1. Clone source ===
cd ~/Applications || mkdir -p ~/Applications && cd ~/Applications

if [ -d App-LabRecorder ]; then
  echo "🔁 Reusing existing repo: App-LabRecorder"
else
  git clone https://github.com/labstreaminglayer/App-LabRecorder.git
fi

cd App-LabRecorder

# === 2. Configure build ===
echo "⚙️ Configuring CMake..."

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=${PWD}/install \
  -DBOOST_ROOT=/opt/homebrew \
  -DBoost_INCLUDE_DIR=/opt/homebrew/include \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DCMAKE_PREFIX_PATH=$(brew --prefix qt@5)

# === 3. Build ===
echo "🔨 Building LabRecorder..."
cmake --build build --config Release -j8 --target install

# === 4. Fix Qt plugin paths ===
echo "🧩 Fixing Qt plugin paths..."
install_name_tool -change @rpath/QtGui.framework/Versions/A/QtGui \
  @executable_path/../Frameworks/QtGui \
  install/LabRecorder/LabRecorder.app/Contents/PlugIns/platforms/libqcocoa.dylib

# === 5. Code signing ===
echo "🔏 Applying lightweight codesign..."
APP="install/LabRecorder/LabRecorder.app"

codesign --remove-signature "$APP" || true

find "$APP/Contents/Frameworks" -type f -exec codesign --force --sign - {} \; 2>/dev/null
find "$APP/Contents/PlugIns" -type f -exec codesign --force --sign - {} \; 2>/dev/null
codesign --force --deep --sign - "$APP"

# === 6. Final check ===
echo "✅ Build complete! Launching LabRecorder..."
open "$APP"

echo "📁 You can find the app at: $PWD/install/LabRecorder/LabRecorder.app"