#!/bin/bash
set -e

echo "Installing Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs

echo "Pulling LFS files..."
git lfs install
git lfs pull

echo "Build complete!"
