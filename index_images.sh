. .venv/bin/activate
. .env

python util/load_bottles.py --model-name vgg16
python util/load_bottles.py --model-name vit16
