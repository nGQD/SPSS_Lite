# SPSS_Lite
A statistical calculator developed by nGQD for satisfaction.
It is currently able to compute Linear Regression and One-Way ANOVA Test for 2 variables/categories. The computed statistics would be wrapped inside pandas DataFrame and converted into latex, which would be compiled by [MiKTeX](https://miktex.org/download). The PDF would get embedded within an HTML and prompted upon every execution for the ease of inspection (there's a visualization in the HTML too btw).

You can always modify the codes to your desired format by checking out stats/template.txt.

## Installation

0. (Optional) Setup and activate venv
1. Run pip install -r requirements.txt on cmd
2. Install MiKTeX
3. You're good to go!

## Output
<iframe src="./stats/latex.pdf" style="width: 100%;height: 100%;border: none;"></iframe>
