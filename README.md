# Awesome Active Learning [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A list of resources related to Active learning in machine learning.

|  Title  | Venue |
|:--------|:--------:|
|[Deep Active Learning for Biased Datasets via Fisher Kernel Self-Supervision](http://openaccess.thecvf.com/content_CVPR_2020/papers/Gudovskiy_Deep_Active_Learning_for_Biased_Datasets_via_Fisher_Kernel_Self-Supervision_CVPR_2020_paper.pdf)|CVPR'20|
|[State-Relabeling Adversarial Active Learning](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_State-Relabeling_Adversarial_Active_Learning_CVPR_2020_paper.pdf)|CVPR'20|
|[Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds](https://openreview.net/pdf?id=ryghZJBKPS) | ICLR'20 |
|[Adversarial Sampling for Active Learning](https://arxiv.org/abs/1808.06671) | WACV'20 |
|[Discriminative Active Learning](https://arxiv.org/pdf/1907.06347.pdf)| arXiv'19 |
|[Semantic Redundancies in Image-Classification Datasets: The 10% You Don’t Need](https://arxiv.org/pdf/1901.11409.pdf)| arXiv'19 |
|[BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning](http://papers.nips.cc/paper/8925-batchbald-efficient-and-diverse-batch-acquisition-for-deep-bayesian-active-learning.pdf) | NIPS'19|
|[Bayesian Batch Active Learning as Sparse Subset Approximation](http://papers.nips.cc/paper/8865-bayesian-batch-active-learning-as-sparse-subset-approximation.pdf) | NIPS'19 |
| [Learning Loss for Active Learning](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Learning_Loss_for_Active_Learning_CVPR_2019_paper.pdf) | CVPR'19 |
| [Bayesian Generative Active Deep Learning](http://proceedings.mlr.press/v97/tran19a/tran19a.pdf) | ICML'19 |
| [Variational Adversarial Active Learning](http://openaccess.thecvf.com/content_ICCV_2019/papers/Sinha_Variational_Adversarial_Active_Learning_ICCV_2019_paper.pdf) | ICCV'19 |
| [Active Learning for Convolutional Neural Networks: A Core-Set Approach](https://openreview.net/pdf?id=H1aIuk-RW) | ICLR'18|
| [The Power of Ensembles for Active Learning in Image Classification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf) | CVPR'18 |
|[Adversarial Active Learning for Sequence Labeling and Generation](https://www.ijcai.org/proceedings/2018/0558.pdf) | IJCAI'18 |
| [Active Decision Boundary Annotation with Deep Generative Models](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huijser_Active_Decision_Boundary_ICCV_2017_paper.pdf) | ICCV'17|
| [Deep Bayesian Active Learning with Image Data](http://proceedings.mlr.press/v70/gal17a/gal17a.pdf) | ICML'17|
| [Generative Adversarial Active Learning](https://arxiv.org/pdf/1702.07956.pdf) | arXiv'17 |
|[Active Image Segmentation Propagation](http://openaccess.thecvf.com/content_cvpr_2016/papers/Jain_Active_Image_Segmentation_CVPR_2016_paper.pdf)| CVPR'16 |
|[Cost-Effective Active Learning for Deep Image Classification](https://arxiv.org/pdf/1701.03551.pdf) | TCSVT'16 |


# Tutorials

* [[Book] Active Learning](https://www.morganclaypool.com/doi/abs/10.2200/S00429ED1V01Y201207AIM018). Burr Settles. (CMU, 2012)
* [[Youtube] Active Learning from Theory to Practice](https://www.youtube.com/watch?v=_Ql5vfOPxZU). Steve Hanneke, Robert Nowak. (ICML, 2019)



# Papers

## Pool-Based Sampling

### Singleton

* [A Variance Maximization Criterion for Active Learning](https://arxiv.org/pdf/1706.07642.pdf). Yazhou Yang, Marco Loog. (Pattern Recognition, 2018)

* [The power of ensembles for active learning in image classification](http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf). William H. Beluch, Tim Genewein, Andreas Nurnberger, Jan M. Kohler. (CVPR, 2018)

* [Learning Algorithms for Active Learning](https://arxiv.org/pdf/1708.00088.pdf). Philip Bachman, Alessandro Sordoni, Adam Trischler. (ICML, 2017)

* [Learning Active Learning from Data](https://papers.nips.cc/paper/7010-learning-active-learning-from-data.pdf). Ksenia Konyushkova, Sznitman Raphael. (NIPS, 2017)

* [Beyond Disagreement-based Agnostic Active Learning](https://papers.nips.cc/paper/5435-beyond-disagreement-based-agnostic-active-learning.pdf). Chicheng Zhang, Kamalika Chaudhuri. (NIPS, 2014)

* [Active Learning using On-line Algorithms](https://www.cs.rutgers.edu/~pazzani/Publications/active-online.pdf). Chris Mesterharm, Michael J. Pazzani. (KDD, 2011) 

* [Active Learning from Crowds](http://www.cs.columbia.edu/~prokofieva/CandidacyPapers/Yan_AL.pdf). Yan Yan, R ́omer Rosales, Glenn Fung, Jennifer G. Dy. (ICML, 2011)

* [Hierarchical Sampling for Active Learning](https://dl.acm.org/doi/pdf/10.1145/1390156.1390183). Sanjoy Dasgupta, Daniel Hsu  (ICML, 2008)

  

### Batch/Batch-like

* [Variational Adversarial Active Learning](https://arxiv.org/pdf/1904.00370.pdf). Samarth Sinha, Sayna Ebrahimi, Trevor Darrell. (ICCV, 2019)
* [Integrating Bayesian and Discriminative Sparse Kernel Machines for Multi-class Active Learning](https://papers.nips.cc/paper/8500-integrating-bayesian-and-discriminative-sparse-kernel-machines-for-multi-class-active-learning.pdf). Weishi Shi, Qi Yu. (NeurIPS, 2019)
* [Rapid Performance Gain through Active Model Reuse](http://www.lamda.nju.edu.cn/liyf/paper/ijcai19-acmr.pdf). Feng Shi, Yu-Feng Li. (IJCAI, 2019)
* [Meta-Learning for Batch Mode Active Learning](https://openreview.net/forum?id=r1PsGFJPz). Sachin Ravi, Hugo Larochelle. (ICLR-WS, 2018)
* [Active Semi-Supervised Learning Using Sampling Theory for Graph Signals](http://sipi.usc.edu/~ortega/Papers/Gadde_KDD_14.pdf). Akshay Gadde, Aamir Anis, Antonio Ortega. (KDD, 2014)
* [Active Learning for Multi-Objective Optimization](http://proceedings.mlr.press/v28/zuluaga13.pdf). Marcela Zuluaga, Andreas Krause, Guillaume Sergent, Markus P{\''u}schel (ICML, 2013)
* [Querying Discriminative and Representative Samples forBatch Mode Active Learning](http://chbrown.github.io/kdd-2013-usb/kdd/p158.pdf). Zheng Wang, Jieping Ye. (KDD, 2013)
* [Near-optimal Batch Mode Active Learning and Adaptive Submodular Optimization](http://proceedings.mlr.press/v28/chen13b.pdf). Yuxin Chen, Andreas Krause. (ICML, 2013)
* [Active Learning for Probabilistic Hypotheses Usingthe Maximum Gibbs Error Criterion](https://papers.nips.cc/paper/4958-active-learning-for-probabilistic-hypotheses-using-the-maximum-gibbs-error-criterion.pdf), Nguyen Viet Cuong, Wee Sun Lee, Nan Ye, Kian Ming A. Chai, Hai Leong Chieu. (NIPS, 2013)
* [Batch Active Learning via Coordinated Matching](https://icml.cc/2012/papers/607.pdf). Javad Azimi, Alan Fern, Xiaoli Z. Fern, Glencora Borradaile, Brent Heeringa. (ICML, 2012)
* [Ask me better questions: active learning queries based on rule induction](https://www.eecs.wsu.edu/~cook/pubs/kdd11.pdf). Parisa Rashidi, Diane J. Cook. (KDD, 2011)
* [Active Instance Sampling via Matrix Partition](https://papers.nips.cc/paper/3919-active-instance-sampling-via-matrix-partition). Yuhong Guo. (NIPS 2010)
* [Discriminative Batch Mode Active Learning](https://papers.nips.cc/paper/3295-discriminative-batch-mode-active-learning.pdf). Charles X. Ling, Jun Du. (NIPS, 2007)



## Stream-Based Selective Sampling

* [Active Learning from Peers](https://papers.nips.cc/paper/7276-active-learning-from-peers.pdf). Keerthiram Murugesan, Jaime Carbonell. (NIPS, 2017)
* [An Analysis of Active Learning Strategies for Sequence Labeling Tasks](https://www.biostat.wisc.edu/~craven/papers/settles.emnlp08.pdf). Burr Settles, Mark Craven. (EMNLP, 2008)
* [Improving Generalization with Active Learning](https://users.cs.northwestern.edu/~pardo/courses/mmml/papers/active_learning/improving_generalization_with_active_learning_ML94.pdf), DAVID COHN, LES ATLAS, RICHARD LADNER. (Machine Learning, 1994)



## Membership Query Synthesize

* [Active Learning via Membership Query Synthesisfor Semi-supervised Sentence Classification](https://www.aclweb.org/anthology/K19-1044/). Raphael Schumann, Ines Rehbein. (CoNLL, 2019)
* [Active Learning with Direct Query Construction](https://dl.acm.org/doi/10.1145/1401890.1401950), Yuhong Guo, Dale Schuurmans. (KDD, 2008)



## Others

