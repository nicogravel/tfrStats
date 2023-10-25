# tfrStats

a mini-tutorial on assessing the statistical significance of a Time Frequency Representation (TFR)


[https://nicogravel.github.io/tfrStats/](https://nicogravel.github.io/tfrStats/html/index.html)


# <span style="color:#3498db">**A mini-tutorial**</span>

This codebook is a simple example meant to illustrate a two fundamental questions in time frequency analysis: 1) permutation-based null hypothesis testing and 2) correction for multiple comparisons. Typically, there is more than one approach for a given scenario. They will all depend on the dimensions of the data at hand, be these spatial locations, time and frequency analysis parameters, trials of different conditions, the nature of hypothesis, etc. For basic pedagogical purposes, here I focus on spectral power increases relative to baseline using two variants of the same method: 1) null hypothesis testing using the traditional min-max distribution approach, which captures variations at the extremes of the null ditribution, and 2) null hypthesis testing using the whole null distribution, obtained by averaging across specific dimensions. I invite the reader to audit the code and propvide feedback and comments in the open discussion subsection within the background section. Since there are several ways to achieve these goals and many realizations of these and other related methods (i.e. thresholds may be obtaind from the percentiles of the null distribution directly and further corrected in equivalent ways, or the pooling of data accomplished among specific dimensions), here I focus on two common methods using very simple examples in the hope to help those researchers (including myself) that are or may be in need of clarity on these matters, touch ground. Beware there may even be error to be spot or loops to optimize. 

Depending on the approach, the computation of the p-values will change slightly. In the min-max approach the minimum and maximum values at each permutations are used. When testing using the whole null distribution, the null values obtained from the permutations are averaged the same way the empirical distribution is averaged so the dimensions of the empirical distributions are preserved in the null distribution. Once the null distributions have been obtained, p-values are obtained using the empirical cumulative distribution method. Additionally, an optional step is illustrated in which cluster correction of the p-values is implemented. Provided the right Python environment is installed and data, this Jupyter notebook should work as a simple mini-tutorial and support the discussion of these and related basic approaches for computing TFRs and assessing their statistical significance in a clear way. The functions within this notebook are provided in an exploratory state and are subject to change. A background section with an open discussion subsection is included. Please feel free to use it help improve the codebook. The package can be downloaded from here: https://github.com/nicogravel/tfrStats






## Installation

To run the [notebook](https://github.com/nicogravel/tfrStats/blob/main/docs/html/notebooks/statistical_approach.ipynb), clone the package (or your fork) locally and then:
  
    
    
```
git clone https://github.com/nicogravel/tfrStats.git

cd tfrStats

conda env create --name tfrStats-dev --file tfrStats.yml

conda activate tfrStats-dev

pip install -e .
```
  
    
    

Voilà!