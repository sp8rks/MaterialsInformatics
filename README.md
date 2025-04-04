# MaterialsInformatics
MSE5540/6640 Materials Informatics course at the University of Utah

This github repo contains coursework content such as class slides, code notebooks, homework assignments, literature, and more for MSE 5540/6640 "Materials Informatics" taught at the University of Utah in the Materials Science & Engineering department. 

Below you'll find the approximate calendar for Spring 2025 and videos of the lectures are being placed on the following YouTube playlist
https://youtube.com/playlist?list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0

![My Image](YT_playlist.jpg)


| month | day | Subject to cover                                                                          | Assignment                                                                                                                  | Link                          |
|-------|-----|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Jan   | 7  | Syllabus. What is machine learning? How are materials discovered?   Machine Learning vs Materials Informatics                      |  Install software packages together in class                                                                                                                           |                               |
| Jan   | 9  | Using Github, In class example of fitting Hall-Petch data with linear model                    | Read 5 High Impact Research Areas in ML for MSE (paper1), Read ISLP Chapter 3, but especially Section 3.1                                  | [paper1](https://doi.org/10.1021/acs.chemmater.9b04078), [ISLP](https://www.statlearning.com/)               |
| Jan   | 14  | Materials data repositories, get pymatgen running for everybody, examples of MP API, MDF, NOMAD, others           | Create a new env and make sure you can get the notebooks in the "worked examples/MP_API_example" and "worked examples/foundry" folders running. | [Materials Project API](https://next-gen.materialsproject.org/api)                      |
| Jan   | 16  | Machine Learning Tasks and Types, Featurization in ML, Composition-based feature vector   | Read Is domain knowledge necessary for MI (paper1). Make sure you can get the CBFV_example notebook running in the ""worked examples/CBFV_example" folder                                                                          | [paper1](https://doi.org/10.1007/s40192-020-00179-z)                       |
| Jan   | 21  | Best Practices introduction and Classification | Read ISLP Sections 4.1-4.5 and Section 5.1. Read Best Practices paper. Run through classification notebooks| [ISLP](https://www.statlearning.com/), [classification paper](https://doi.org/10.1021/acs.chemmater.7b05304)|
| Jan   | 23  | Structure-based feature vector, crystal graph networks, SMILES vs SELFIES, 2pt statistics | read selfies (paper1), two-point statistics (paper2) and intro to graph networks (blog1)                                    | [paper1](https://doi.org/10.1088/2632-2153/aba947), [paper2](https://linkinghub.elsevier.com/retrieve/pii/S1359645408004886), [blog1](https://distill.pub/2021/gnn-intro/)         |
| Jan   | 28  | Simple linear/nonlinear models. test/train/validation/metrics                             | Read linear vs non-linear (blog1), read benchmark overfitting (paper1), and loco-cv (paper2). | [blog1](https://statisticsbyjim.com/regression/choose-linear-nonlinear-regression/), [paper1](https://www.nature.com/articles/s41524-023-01012-9), [paper2](https://doi.org/10.1039/C8ME00012C) |
| Jan   | 30  | in-class examples of featurization                             | Run through 2pt statistics, RDKit notebooks |HW1 due!  |
| Feb   | 4   | ensemble models, ensemble learning                                                  | Read ensemble (blog1), and ensemble learning (paper1)                                                                                       |[blog1](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205),  [paper1](https://doi.org/10.1007/s40192-020-00178-0)                  |
| Feb   | 6   | Extrapolation, support vector machines, clustering                                              | Read extrapolation to extraordinary materials (paper1), clustering (paper2) , SVMs (blog2)                     | [paper1](https://doi.org/10.1016/j.commatsci.2019.109498), [paper2](https://www.nature.com/articles/s42003-022-03628-x),  [blog2](https://towardsdatascience.com/the-complete-guide-to-support-vector-machine-svm-f1a820d8af0b)      |
| Feb   | 11   | Artificial neural networks                                                                | Read the introduction to neural networks (blog1) and math of neural networks (blog2)                                                                     | [blog1](https://victorzhou.com/blog/intro-to-neural-networks/), [blog2](https://medium.com/towards-data-science/introduction-to-math-behind-neural-networks-e8b60dbbdeba)                  |
| Feb   | 13  | Advanced deep learning (CNNs, RNNs)                                                       | HW2 due. Read…                                                                                                              | [blog1](https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns), [blog2](https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/)                  |
| Feb   | 18  | Transformers                                                                              | Read the introduction to transformers (blog1, blog2)                                                                        | [blog1](https://www.datacamp.com/tutorial/how-transformers-work), [blog2](https://www.columbia.edu/~jsl2239/transformers.html)                  |
| Feb   | 20  | Generative ML: Generative Adversarial Networks and variational autoencoders               | Read about VAEs (blog1, blog2, repo1) and GANS (blog3)                                                                           | [blog1](https://visualstudiomagazine.com/articles/2021/05/06/variational-autoencoder.aspx?m=1), [blog2](https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/), [repo1](https://github.com/AntixK/PyTorch-VAE), [blog3](https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/)           |
| Feb    | 25  | Diffusion models and Image segmentation| Read U-net (paper1) and nuclear forensics (paper2)                                                                                                                         |                         |
| Feb    | 27  | Image segmentation part 2 and in-class coding examples | Download CrysTens github repo, read Segment Anything Model (paper 3)                                                                                                                         |   [paper1](https://arxiv.org/pdf/1505.04597.pdf), [paper2](https://doi.org/10.1016/j.jnucmat.2019.01.042), [paper3](https://arxiv.org/abs/2304.02643)                            |
| Mar   | 4  | Diffusion models                                                                       |                                                                           |     [CrysTens repo](https://github.com/michaeldalverson/CrysTens)                      |
| Mar   | 6  |  Bayesian Inference                        |   Read the introduction to Bayesian (blog1), go through Naive Bayes notebook, HW 3 due!    |                              |
| Mar   | 11  | Gaussian Processes                                                                        |Read blog 1    | [blog1](https://distill.pub/2019/visual-exploration-gaussian-processes/)                        |
| Mar   | 13  |No CLASS, spring break |||
| Mar   | 19  |No CLASS, spring break                       |       |                              |
| Mar   | 18  | Live coding Bayesian Optimization   |   |      |
| Mar   | 20  | Large Language Models part 1                       |       |                              |
| Mar   | 25   | No CLASS, TMS Meeting|||
| Mar   | 27   | NO CLASS, TMS Meeting   |||
| Apr   | 1   | Large Language Models part 2                                                          | TBD                                                                                                                         | TBD                           |
| Apr   | 3  | Case study TBD|||
| Apr   | 8  | Case study TBD|||
| Apr   | 15  | Case study TBD|||
| Apr   | 17  | Case study TBD|||
| Apr   | 22  | Final project presentation|||

|       |     |                                                                                           |                                                                                                                             |                               |


I can recommend the book Introduction to Machine Learning found here https://www.statlearning.com/
