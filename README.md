# tfrStats

a mini-tutorial on assessing the statistical significance of a Time Frequency Representation (TFR)


[https://nicogravel.github.io/tfrStats/](https://nicogravel.github.io/tfrStats/html/index.html)


# <span style="color:#3498db">**A mini-tutorial**</span>

This codebook is a simple example meant to illustrate a basic question on time frequency analysis: assessing the significance of increases in spectral power using permutation statistics and nul distribution testing. As the reader may know, there is a variety of approaches aimed at correcting for multiple comparisons and reduce false positives in different scenarios. They will all depend on the dimensions of the data at hand, be these spatial locations, time and frequency windows or trials of different conditions, as well as the nature of hypothesis.

For basic pedagogical purposes, here I focus on spectral power increases relative to baseline using two variants of the same method: 1) null hypothesis testing using the traditional min-max distribution approach, which captures the variations at the extreme of the null ditribution, and 2) null hypthesis testing using the whole null distribution. I invite the reader to audit the code and propvide feedback and comments in the open discussion subsection within the backgrond section. 

Depending on the approach, the computation of the p-values will change slightly. In the min-max approach the minimum and maximum values at each permutations are used. When testing using the whole null distribution, the null values obtained from the permutations are averaged the same way the empirical distribution is averaged so the dimensions of the empirical distributions are preserved in the null distribution. Once the null distributions have been obtained, p-values are obtained using the empirical cumulative distribution method.

An optional step is illustrated in which cluster correction of the p-values is implemented. Since there are several ways to achieve these goals and many differences exist among the many realizations of these and other related methods (i.e. thresholds may be obtaind from the percentiles of the null distribution directly and further corrected in equivalent ways, or the pooling of data for multiple comparison correction accomplished among different dimensions through averaging, normalization, etc), here I focus on two common methods using very simple examples in the hope to help those researchers (including myself) that are or may be in need of clarity on these matters, touch ground. 

Provided the right Python environment and data, this Jupyter notebook should work as a simple mini-tutorial to help discuss basic approaches for computing TFRs and assess their statistical significance in a *toolbox free*, transparent, barebones DIY way. The functions within this notebook are provided in an exploratory state and are subject to change. A background section with an open discussion subsection is included. Please feel free to use it help improve the codebook. Optionally, the package can be downloaded from here: https://github.com/nicogravel/tfrStats
