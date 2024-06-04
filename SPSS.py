import numbers
import subprocess
import webbrowser as wb
from math import sqrt
import os

import numpy as np
import pandas as pd
import plotly.express as px
from numpy.core.fromnumeric import std
from scipy import stats
from sklearn.linear_model import LinearRegression

os.environ["PATH"] += os.pathsep + r"C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64"


class SPSS:
    
    "Fake SPSS created by nGQD"

    def __init__(self) -> None:
        self.model = LinearRegression()

    @property
    def model(self) -> LinearRegression:
        return self.__model

    @model.setter
    def model(self, model: LinearRegression) -> None:
        self.__model = model

    @property
    def sample_x(self) -> np.ndarray:
        return self.__sample_x

    @property
    def sample_y(self) -> np.ndarray:
        return self.__sample_y

    @sample_x.setter
    def sample_x(self, sample_x: np.ndarray) -> None:
        self.__sample_x = sample_x.reshape(-1 ,1)

    @sample_y.setter
    def sample_y(self, sample_y: np.ndarray) -> None:
        self.__sample_y = sample_y

    def fit(self) -> None:
        self.model.fit(self.sample_x, self.sample_y)

    def predict(self, x: numbers.Number) -> numbers.Number:
        return self.model.predict(np.array(x).reshape(-1, 1))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x.reshape(-1, 1))


    def model_summary(self) -> pd.DataFrame:

        "Return DataFrame containing Model Summary"

        df = pd.DataFrame([{
                "R" : (r := sqrt(self.model.score(self.sample_x, self.sample_y))),
                "R-squared" : (rsq := r ** 2),
                "Adjusted R-squared" : (arsq := 1 - ((1-rsq) * ((n := self.sample_x.size)-1) / (n-2))),
                "Std Err of Estimate" : sqrt(1-arsq) * std(self.sample_y, ddof=1)
        }])
        return df
    

    def anova(self) -> pd.DataFrame:

        "Return DataFrame containing ANOVA analysis"

        df = pd.DataFrame({
                "Model" : ["F-Statistic", "P-value"],
                "One-Way" : [
                    *stats.f_oneway(self.sample_x.flatten(), self.sample_y),
                ],
            })
        return df
       

    def coefficients(self) -> pd.DataFrame:

        "Return DataFrame containing coefficient analysis"

        df = pd.DataFrame({
                "Model" : ["Constant (x)", "Independent Var (y)"],
                "Un-std B" : [
                    b := (linreg := stats.linregress(self.sample_x.flatten(), self.sample_y)).intercept,
                    m := linreg.slope
                ],
                "Std Err" : [
                    b_err := linreg.intercept_stderr,
                    m_err := linreg.stderr
                ],
                "Beta" : [
                    None,
                    std(self.sample_x, ddof=1) / std(self.sample_y, ddof=1) * m
                ],
                "T" : [
                    t_x := b / b_err,
                    t_y := m / m_err
                ],
                "Sig Lvl" : [
                    stats.t.sf(t_x, (dfres := self.sample_x.size-2)) * 2,
                    stats.t.sf(t_y, dfres) * 2
                ]
            })
        return df
        

    def compile_latex(self, filename: str) -> None:
        with open("stats\\template.txt", "r") as f:
            latex = "".join(f.readlines())
            latex += self.model_summary().to_latex(
                        index=False,
                        column_format="cccc",
                        float_format="%.6f",
                        position="h"
                    )
            latex += "\n\\subsection{ANOVA}\n"
            latex += self.anova().to_latex(
                        index=False,
                        column_format="cccccc",
                        float_format="%.6f",
                        position="h",
                        na_rep=""
                    )
            latex += "\n\\subsection{Coefficients}\n"
            latex += self.coefficients().to_latex(
                        index=False,
                        column_format="cccccc",
                        float_format="%.6f",
                        position="h",
                        na_rep=""
                    )
            latex += "\n\n\end{document}"
            with open(filename, "w") as g:
                g.write(latex)

    def plot(self) -> None:
        graph = px.scatter(
            pd.DataFrame(
                {"Constant":self.sample_x.flatten(), "Independent Variable":self.sample_y}
            ),
            title = "Interactive Linear Regression Graph",
            x = "Constant",
            y = "Independent Variable",
            trendline = "ols",
            trendline_color_override="gold",
            template = "presentation",
            color = "Constant",
            color_continuous_scale = "purp"
        )
        graph.write_html(os.path.join(os.getcwd(), r"stats/graph.html"))

if __name__ == "__main__":
    spss = SPSS()


    spss.sample_x = np.array([1,9,8,6])
    spss.sample_y = np.array([6,21,24,19])
    spss.fit()



    spss.compile_latex(os.path.join(os.getcwd(), r"stats/latex.tex"))


    subprocess.run(
        [
            'pdflatex',
            os.path.join(os.getcwd(), r"stats/latex.tex"),
            '-interaction=nonstopmode',
            r"-output-directory=stats"
        ], shell=True
    )

    spss.plot()

    wb.open(os.path.join(os.getcwd(), r"stats/Linear Regression Statistical Report.html"), new=2)

    # C:\Users\user\AppData\Local\Programs\MiKTeX\miktex\bin\x64


# spss = SPSS()
# spss.sample_x = np.array([11,9,9,9,8,8,8,6,6,5,5,5,5,5,5,4,4,4,3,3,3])
# spss.sample_y = np.array([26,21,24,21,19,13,19,11,23,15,13,4,18,12,3,11,15,6,13,4,14])
# spss.fit()

# spss.compile_latex(r"stats/latex.tex")

# print(os.path.join(os.getcwd(), r"stats\latex.tex"))

# subprocess.run(
#     ['pdflatex',
#     os.path.join(os.getcwd(), r"stats\latex.tex"),
#     '-interaction=nonstopmode',
#     r"-output-directory=stats"]
# )

# spss.plot()

# wb.open(r"../stats/Linear Regression Statistical Report.html", new=2)