#!/bin/bash
# Build script for CRAM Generator Java Edition

set -e

echo "🧬 CRAM Generator - Java Build"
echo "==============================="

cd "$(dirname "$0")"

# Check Java version
echo "Checking Java version..."
java -version

# Check Maven
echo "Checking Maven..."
mvn -version

# Clean and build
echo ""
echo "Building project..."
mvn clean package -DskipTests

echo ""
echo "✅ Build complete!"
echo ""
echo "Run as CLI:"
echo "  java -jar target/cram-generator-1.0.0.jar validate \"CC1CCC(CC1)(C(=O)O)C(=O)O\""
echo ""
echo "Run as server:"
echo "  java -jar target/cram-generator-1.0.0.jar --server"
echo ""
