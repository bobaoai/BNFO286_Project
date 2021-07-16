# The project result becomes a part of the paper 
# Carlin, D.E., Fong, S.H., Qin, Y., Jia, T., Huang, J.K., Bao, B., Zhang, C. and Ideker, T., 2019. A fast and flexible framework for network-assisted genomic association. Iscience, 16, pp.155-161.
Slides Link

https://docs.google.com/presentation/d/1to310NDKKWqgWbuMVb4zgLRPSoP3ZXkGajnbKoFuB2I/edit?usp=sharing

Overleaf
https://www.overleaf.com/read/prjnjmhbwxyp
https://www.overleaf.com/16700835nghxcnnkxwtm#/64031895/

## Ranking with sparse NN

  SNPs' statistic scores are obtained.

  NN is built with following structure:

  SNPs -> Gene -> Weighted Gene -> SNPs

  SNPs are sampled for NN trainning and validation

  The Weighted Gene layer is taken out and the weight is used for gene ranking. Done.

## Ranking with Network Propagation

  Use the top-hitted SNPs pvalue as the gene value.

  Run the GWAB or NetWAS to do the network propagation.


## Ranking
