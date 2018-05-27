# BNFO286
### Bokan Bao, Chao Zhang, Yue Qin
Doc Link
https://docs.google.com/presentation/d/1to310NDKKWqgWbuMVb4zgLRPSoP3ZXkGajnbKoFuB2I/edit?usp=sharing

Slides Link

https://docs.google.com/document/d/1Vl2MsCqqYmXSgawZSrlfWuU_cUNVinBiRAVbBI8k1pk/edit?usp=sharing

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
