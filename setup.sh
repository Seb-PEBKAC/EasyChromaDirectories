# under active development

#python3.10 -m venv venv
#source venv/bin/activate
pip install -r requirements.txt
#pip install -r requirements-dev.txt
#pytest test_TryChromav3.py -v

python Chromav4_Encode_documents.py assets_ChromaDB_Vec/

#python Chromav4_Encode_documents.py your_directory/ --list

#python Chromav4_Encode_documents.py your_directory/ --query "Who's the _____ uncle" --n_results 1

#python Chromav4_Encode_documents.py your_directory/ --query "Find similar images and text about nature" --n_results 1
