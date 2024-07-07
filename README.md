# SENAI
make sure you have ollama and llama3 installed locally

Download the whole zip
# Create the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the required modules
pip install -r requirements.txt

#Update the location of the csv file
then use pwd to find the location of the csv file and change it in pythoncode

#Run the program
chainlit run app.py
