# In the notebook over here ...
# git clone https://github.com/arunpatwa/Text_Summarizer.git
# Text-Summarizer/Fine_Tune_LLM/script.sh
echo -------- Loading Please wait ----------
echo ---------------------------------------
echo ---------------------------------------

# clone the repository over here ...
pip install gdown
pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
pip install transformers[torch] -U
pip install accelerate -U
pip install torch
pip install transformers
pip install requests
pip install tqdm


# gdown --id 1pjs47ye8j3YtMnw0bZi_nX_0d1HHqzzo
# unzip /content/LLumo.zip
# python3 /content/Text-Summarizer/Fine_Tune_LLM/start.py
# python3 /content/Text-Summarizer/Fine_Tune_LLM/Run.py
# # code for running the code ...

# # code for downloading the zip model over here
# gdown --id 1pjs47ye8j3YtMnw0bZi_nX_0d1HHqzzo

# def predict_text(input_text):
#     # Your model prediction logic here
#     # For demonstration purposes, let's just print the input text
#     print("User input:", input_text)
#     # write the function over here to predict from the model...





# # Connect the UI to the Python function
# from google.colab import output

# def run_prediction(input_text):
#     predict_text(input_text)


# output.register_callback('notebook.runPrediction', run_prediction)

# from IPython.display import HTML, display
# ui_code = '''
# <script>
#   function predict() {
#     // Get the input value
#     var userInput = document.getElementById('inputText').value;

#     // Send the input to Colab Python code for processing
#     google.colab.kernel.invokeFunction('notebook.runPrediction', [userInput], {});
#   }
# </script>

# <div>
#   <label for="inputText">Enter Text:</label>
#   <input type="text" id="inputText" placeholder="Type your text here...">
#   <button onclick="predict()">Predict</button>
# </div>
# '''

# display(HTML(ui_code))




