git lfs install
git lfs pull
source ./.venv/bin/activate
pip3 install -r --force-reinstall requirements.txt
python -m spacy download it_core_news_lg
python -m spacy download es_dep_news_trf
