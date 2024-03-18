from IPython.display import HTML, display
from google.colab import output

import sys
sys.path.append('/content/Text-Summarizer')

from Run import TextSummarizerUI
text_summarizer_ui = TextSummarizerUI()
text_summarizer_ui.diPlay()

output.register_callback('notebook.runPrediction', run_prediction)


