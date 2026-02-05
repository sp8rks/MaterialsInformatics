# MaterialsInformatics
MSE5540/6640 Materials Informatics course at the University of Utah

This github repo contains coursework content such as class slides, code notebooks, homework assignments, literature, and more for MSE 5540/6640 "Materials Informatics" taught at the University of Utah in the Materials Science & Engineering department.

Below you'll find the approximate calendar for Spring 2026 and videos of the lectures are being placed on the following YouTube playlist:  
[YouTube playlist](https://youtube.com/playlist?list=PLL0SWcFqypCl4lrzk1dMWwTUrzQZFt7y0)

![My Image](YT_playlist.jpg)

| month | day  | Subject to cover | Readings | Code/Notebooks | Assignment |
|------:|:----:|------------------|----------|----------------|------------|
| Jan   | 6    | Syllabus, What is ML, Materials discovery |  | Install software packages |  |
| Jan   | 8    | Using Github, Hall-Petch fitting | Read 5 High Impact Research Areas in ML for MSE ([paper](https://doi.org/10.1021/acs.chemmater.9b04078))<br>Read ISLP Chapter 3 Section 3.1 ([ISLP](https://www.statlearning.com/)) |  |  |
| Jan   | 13   | Materials data repositories, pymatgen, MP API | [Materials Project API](https://next-gen.materialsproject.org/api) | MP_API_example, foundry notebooks |  |
| Jan   | 15   | ML Tasks and Types, Featurization, CBFV | Read domain knowledge paper ([paper](https://doi.org/10.1007/s40192-020-00179-z)) | CBFV_example notebook |  |
| Jan   | 20*  | Best Practices and Classification | Read ISLP Sections 4.1-4.5, 5.1 ([ISLP](https://www.statlearning.com/))<br>Best Practices paper ([paper](https://doi.org/10.1021/acs.chemmater.0c01907)) | Classification notebooks | HW1 out |
| Jan   | 22*  | Structure-based feature vector, crystal graphs, SMILES/SELFIES, 2pt statistics | Selfies paper ([paper](https://doi.org/10.1088/2632-2153/aba947))<br>Two-point statistics paper ([paper](https://linkinghub.elsevier.com/retrieve/pii/S1359645408004886))<br>Intro to graph networks ([blog](https://distill.pub/2021/gnn-intro/)) |  |  |
| Jan   | 27   | Linear/nonlinear models, test/train/validation | Linear vs non-linear ([blog](https://statisticsbyjim.com/regression/choose-linear-nonlinear-regression/))<br>Benchmark dataset paper ([paper](https://doi.org/10.1038/s41524-020-00406-3))<br>LOCO-CV paper ([paper](https://doi.org/10.1039/C8ME00012C)) |  |  |
| Jan   | 29   | Featurization in-class coding + case study |  | 2pt statistics, RDKit notebooks | |
| Feb   | 3*   | Ensemble models and learning | Ensemble methods ([blog](https://towardsdatascience.com/ensemble-methods-bagging-boosting-and-stacking-c9214a10a205))<br>Ensemble learning paper ([paper](https://doi.org/10.1007/s40192-020-00178-0)) |  | **HW1 due!**  |
| Feb   | 5*   | Extrapolation, SVMs, clustering | Extrapolation paper ([paper](https://doi.org/10.1016/j.commatsci.2019.109498))<br>Clustering/UMAP explainer ([blog](https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668))<br>SVM guide ([blog](https://towardsdatascience.com/the-complete-guide-to-support-vector-machine-svm-f1a820d8af0b)) |  | HW2 out |
| Feb   | 10   | Case Study TBD + **Paper Forum I** |  |  |  |
| Feb   | 12*  | Artificial neural networks | Intro to neural networks ([blog](https://towardsdatascience.com/machine-learning-for-beginners-an-introduction-to-neural-networks-d49f22d238f9))<br>Neural networks series ([blog](https://towardsdatascience.com/a-gentle-introduction-to-neural-networks-series-part-1-2b90b87795bc)) |  |  |
| Feb   | 17*  | Advanced deep learning (CNNs, RNNs) | CNNs guide ([blog](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53))<br>RNNs blog (link TBD) |  |  |
| Feb   | 19*  | Transformers | What is a transformer? ([blog](https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04))<br>Illustrated transformers guide ([blog](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)) |  | **HW2 due!** |
| Feb   | 24*  | Generative ML (GANs, VAEs) | VAE overview ([blog](https://visualstudiomagazine.com/articles/2021/05/06/variational-autoencoder.aspx?m=1))<br>VAE in PyTorch ([blog](https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/))<br>PyTorch-VAE repo ([repo](https://github.com/AntixK/PyTorch-VAE))<br>U-net paper ([paper](https://arxiv.org/pdf/1505.04597.pdf))<br>Nuclear forensics paper ([paper](https://doi.org/10.1016/j.jnucmat.2019.01.042)) |  | HW3 out |
| Feb   | 26   | Diffusion models part 1 | Segment Anything Model ([paper](https://arxiv.org/abs/2304.02643)) | [CrysTens repo](https://github.com/michaeldalverson/CrysTens) |  |
| Mar   | 3    | Diffusion models part 2 + Image segmentation part 1 |  | coding examples |  |
| Mar   | 5    | Image segmentation part 2 |  |  | **HW 3 due!** |
| Mar   | 10   | No CLASS, spring break |  |  |  |
| Mar   | 12   | No CLASS, spring break |  |  |  |
| Mar   | 17*  | Bayesian Inference | Intro to Bayesian / Gaussian processes visual explainer ([blog](https://distill.pub/2019/visual-exploration-gaussian-processes/)) | Naive Bayes notebook |  |
| Mar   | 19*  | Gaussian Processes | Gaussian processes visual explainer ([blog](https://distill.pub/2019/visual-exploration-gaussian-processes/)) |  | Final Project Briefing |
| Mar   | 24   | Bayesian Optimization in-class coding + case study |  |  |  |
| Mar   | 26   | No CLASS, TMS Meeting |  |  |  |
| Mar   | 31   | No CLASS, TMS Meeting |  |  |  |
| Apr   | 2    | Large Language Models part 1 |  |  |  |
| Apr   | 7    | Large Language Models part 2 + Intro to Agentic AI part 1 |  |  |  |
| Apr   | 9    | Intro to Agentic AI part 2 |  |  |  |
| Apr   | 14   | Crash Course: Autonomous Materials Science w/ Self-Driving Labs |  |  |  |
| Apr   | 16   | Case Study TBD + **Paper Forum II** |  |  |  |
| Apr   | 21   | **Final project presentation** |  |  |  |

I can recommend the book *Introduction to Statistical Learning* found here: [https://www.statlearning.com/](https://www.statlearning.com/)
