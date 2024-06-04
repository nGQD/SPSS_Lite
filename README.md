# SPSS_Lite
A statistical calculator side-project developed by nGQD for satisfaction.
It is currently able to compute Linear Regression and One-Way ANOVA Test for 2 variables/categories. The computed statistics would be wrapped inside pandas DataFrame and converted into latex, which would be compiled by [MiKTeX](https://miktex.org/download). The PDF would get embedded within an HTML and prompted upon every execution for the ease of inspection (there's a visualization in the HTML too btw).

You can always modify the codes to your desired format by checking out stats/template.txt.

## Installation

0. (Optional) Setup and activate venv
1. Run pip install -r requirements.txt on cmd
2. Install MiKTeX
3. You're good to go!

## Output
<embed style = "margin: auto; min-height:100vh; width: 100%" src = "./stats/latex.pdf#view=FitH"></embed>
