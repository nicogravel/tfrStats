# tfrStats: a mini-tutorial on assessing the statistical significance of a Time Frequency Representation


[https://nicogravel.github.io/tfrStats/](https://nicogravel.github.io/tfrStats/html/index.html)


Recently I came across one article that highlighted the replication crisis in biology. Why so often, results obtained by different labs using the same data are difficult or even impossible to replicate reliably? According to the article, there is some agreement about what may be partly causing this replication crisis: in many cases statisticians act more like plumbers rather than like a priest. While I personally do not agree with the dogmatic view of priest-like statisticians imposing their hypothesis-making machinery to every problem they stumble upon, even if the nature of the problem lay beyond their field, I do agree on the need to reach some collective consensus, especially when the delicate decisions involving the statistical assessment of given problem may contribute with more confusion than clarity.

While good intentioned, the priest-like perspective should not turn into [*scientificism/scientism*](https://www.merriam-webster.com/dictionary/scientism), *or the urge to trust on the temporary answers our good old metrics provide more than the underlying problem that inspired them in first place. Funnily, at the same time I was reading this article I came across another gem in ––now defunct–– Twitter. [A post](https://twitter.com/lakens/status/1718654122516156777) by [Daniël Lackens](https://twitter.com/lakens) provided the much needed, *so zu sagen*, plumber's perspective. After all, plumbing and fitting are delicate activities whose results can either resemble a pipe jungle or a professional design.

In this mini-tutorial, I show how two approaches, when applied to the same scenario, can lead to the same conclusions. I provide basic Python code to illustrate how two different but fundamentally similar pipelines can lead to slightly different but comparable results. The pipelines are based on examples provided in [Fieldtrip](https://www.fieldtriptoolbox.org/workshop/oslo2019/statistics/#permutation-test-with-cluster-correction) and adapted from the book [*Analyzing Neural Time Series Data: Theory and Practice*](https://direct.mit.edu/books/book/4013/Analyzing-Neural-Time-Series-DataTheory-and)

Specifically, to assess the statistical significance of spectral estimates obtained from electrophysiological data (i.e. LFP) we used non-parametric permutation tests and focused on the multiple comparison correction of time frequency representations (TFRs). The success of the two approaches depends on the dimensions of the data at hand, be these spatial locations, time and frequency analysis parameters, trials of different conditions, the nature of hypothesis, etc. For basic pedagogical purposes, here I focus on spectral power increases relative to baseline using two variants of the same method: 1) null hypothesis testing using the min-max distribution approach, which captures variations at the extremes of the null distribution, and 2) null hypothesis testing using the whole null distribution, obtained by averaging across specific dimensions. Since there are several ways to achieve these goals and many realizations of these and other related methods (i.e. thresholds may be obtained from the percentiles of the null distribution directly and further corrected in equivalent ways, or the pooling of data accomplished among specific dimensions), here I focus on two common methods using very simple examples in the hope to help those researchers (including myself) that are or may be in need of clarity on these matters, touch ground. Beware there may even be error to be spot or loops to optimize.

Depending on the approach, the computation of the p-values will change slightly. In the min-max approach the minimum and maximum values at each permutations are used. When testing using the whole null distribution, the null values obtained from the permutations are averaged the same way the empirical distribution is averaged so the dimensions of the empirical distributions are preserved in the null distribution. Once the null distributions have been obtained, p-values are obtained using the empirical cumulative distribution method.

Provided the right Python environment is installed and data, this Jupyter notebook should work as a simple mini-tutorial and support the discussion of these and related basic approaches for computing TFRs and assessing their statistical significance in a clear and easy to *understand/explain* way. The functions within this notebook are provided in an exploratory state and are subject to change. A background section with an open discussion subsection is included. Please feel free to use it help improve the codebook. The package can be downloaded from here: https://github.com/nicogravel/tfrStats





## Installation

To run the [notebook](https://github.com/nicogravel/tfrStats/blob/main/docs/html/notebooks/statistical_approach.ipynb), clone the package (or your fork) locally, then *cd* to the package folder; create an environment based on the *tfrStats.yml* (do this only once), activate it and then install the package. The package can be edited. To see the changes in the program behaviour after editing the code, reinstall the package and restart python.
  
    
    
```
git clone https://github.com/nicogravel/tfrStats.git

cd tfrStats

conda env create --name tfrStats-dev --file tfrStats.yml

conda activate tfrStats-dev

pip install -e .
```
  
    
    

