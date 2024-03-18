from IPython.display import HTML, display
from transformers import pipeline
class TextSummarizerUI:
    pipe = pipeline('summarization', model = 't5-base')
      
    def __init__(self):
        self.html_code = """
        <!DOCTYPE html>
        <html>

        <head>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 20px;
                }

                .container {
                    max-width: 800px;
                    height: 800px
                    margin: auto;
                    background-color: #fff;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                }

                h1 {
                    text-align: center;
                    color: #333;
                }

                h2 {
                    color: #555;
                }

                #input-text {
                    width: 100%;
                    height: 150px;
                    margin-bottom: 10px;
                }

                #output-container {
                    margin-top: 10px;
                    font-size: 25px;
                    color: black;
                }

                #predict-button {
                    background-color: #4caf50;
                    color: #fff;
                    padding: 10px 15px;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 16px;
                }

                #predict-button:hover {
                    background-color: #45a049;
                }

                #output-summary{
                  color: black;
                  font-weight:bold;
                  font-size:20px;
                }
            </style>
        </head>

        <body>
            <div class="container">
                <h1>Text-Summarizer</h1>
                <h2>Enter the text to summarize:</h2>
                <textarea id="input-text" placeholder="Enter your text here..."></textarea>
                <br>
                <button onclick="predict()" id="predict-button">Summarize</button>
                <div id="output-container">
                    <div id="output-container">
                        <strong>Output:</strong>
                        <p id="output-summary"></p>
                    </div>
                </div>
            </div>

            <script>
                function predict() {
                    var inputText = document.getElementById("input-text").value;
                    google.colab.kernel.invokeFunction('notebook.runPrediction', [inputText], {});
                }
            </script>
        </body>

        </html>
        """

    def diPlay(self):
        display(HTML(self.html_code))






