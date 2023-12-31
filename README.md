# ChromStruct
Knowledge of the three-dimensional structure of chromatin in the cellular nucleus is an important topic of research, as it is connected with physiological and pathological correlates and dysfunctional cell behaviour. As direct observation is at present not feasible, on one side, several experimental techniques have been developed to provide information on the spatial organization of the DNA in the cell; on the other side, several computational methods have been developed to elaborate experimental data and infer 3D chromatin conformations. The most relevant experimental methods are Chromosome Conformation Capture and its derivatives, chromatin immunoprecipitation and sequencing techniques (CHIP-seq), RNA-seq, fluorescence in situ hybridization (FISH) and other genetic and biochemical techniques. All of them provide important and complementary information that relate to the three-dimensional organization of chromatin. However, these techniques employ very different experimental protocols and provide information that is not easily integrated, due to different contexts and different resolutions. Here we provide an open-source tool for inferring the 3D structure of chromatin that, by exploiting a multilevel approach, allows the easy integration of information derived from different experimental protocols and referred to different resolution levels of the structure. The code presented here is an expansion of our previously reported code, ChromStruct. 
Here we deposited three versions of ChromStruct:

- ## ChromStruct 3.1
  In this first version we developed and tested a reconstruction technique that does not require translating contacts into distances, thus avoiding a number of related drawbacks. Also, we introduce a geometrical chromatin chain model that allows us to include sound biochemical and biological constraints in the problem. This model can be scaled at different genomic resolutions, where the structures of the coarser models are influenced by the reconstructions at finer resolutions. The search in the solution space is then performed by a classical simulated annealing, where the model is evolved efficiently through quaternion operators. The presence of appropriate constraints permits the less reliable data to be overlooked, so the result is a set of plausible chromatin configurations compatible with both the data and the prior knowledge. [1]
  
- ## ChromStruct 4.2
  In this improved version we propose a multiscale chromatin model where the chromatin fiber is suitably partitioned at each scale. The partial structures are estimated independently, and connected to rebuild the whole fiber. Our score function consists of a data-fit part and a penalty part, balanced automatically at each scale and each subchain. The penalty part enforces soft geometric constraints. As many different structures can fit the data, our sampling strategy produces a set of solutions with similar scores. The procedure contains a few parameters, independent of both the scale and the genomic segment treated. [2,3]
  
- ## ChromStruct 4.3
  In this version we introduce the possibility of integrating HI-C data with histone mark CHIP-seq, CTCF CHIP-seq and RNA-seq data, browsing between different resolution levels, from few kilobases up to Mega-bases [4]. 


### Refernces:

[1] Caudai, C., Salerno, E., Zoppè, M. et al. Inferring 3D chromatin structure using a multiscale approach based on quaternions. BMC Bioinformatics 16, 234 (2015). https://doi.org/10.1186/s12859-015-0667-0

[2] C. Caudai, E. Salerno, M. Zoppè and A. Tonazzini, "Estimation of the Spatial Chromatin Structure Based on a Multiresolution Bead-Chain Model," in IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 16, no. 2, pp. 550-559, 1 March-April 2019, doi: 10.1109/TCBB.2018.2791439.

[3] C. Caudai, E. Salerno, M. Zoppè, I. Merelli and A. Tonazzini, "ChromStruct 4: A Python Code to Estimate the Chromatin Structure from Hi-C Data," in IEEE/ACM Transactions on Computational Biology and Bioinformatics, vol. 16, no. 6, pp. 1867-1878, 1 Nov.-Dec. 2019, doi: 10.1109/TCBB.2018.2838669.

[4] Caudai, C.; Zoppè, M.; Tonazzini, A.; Merelli, I.; Salerno, E. Integration of Multiple Resolution Data in 3D Chromatin Reconstruction Using ChromStruct. Biology 2021, 10, 338. https://doi.org/10.3390/biology10040338
