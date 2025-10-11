set -eux

rm -rf dist

python -mpip install -U pip
python -mpip install -U build wheel

python -m build --sdist --wheel

echo "Wheel built successfully."
