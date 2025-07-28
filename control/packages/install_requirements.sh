#!/bin/bash

sudo apt update

# necessar packages
packages=(
    git
    build-essential
    libsnmp-dev
)

# install the packages
for pkg in "${packages[@]}"; do
    echo "installing $pkg..."
    sudo apt install -y "$pkg"
done

echo "Done."
